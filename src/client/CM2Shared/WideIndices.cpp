// Reads a submesh's triangle start the way the skin format spells it once it passes 65535 indices.
// Copyright (C) 2026 WarcraftXL
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// A submesh's first triangle index is (level << 16) | indexStart. indexStart alone stops addressing
// at 65535, so that pairing is the only way a skin larger than that can say where a submesh begins.
// Both of the client's index-buffer fills read indexStart as a plain 16-bit field, so every submesh
// past the ceiling sources its triangles from the wrong place -- the geometry is in the file, the
// fill just looks 65536 indices too early.
//
// level also had an older life as a LOD / sub-batch marker, so the fold is conditional: it is taken
// only when the widened start plus the submesh's own index count still lands inside the triangle
// array the skin itself declares. A marker names an offset that array cannot contain, so it is
// rejected and the client's own 16-bit reading stands. Two consequences worth stating: a skin that
// is not writing an extended start is filled by the untouched engine function, and no value of
// level can make the fill read outside the array it is reading from.

#include "common/Log.hpp"
#include "engine/assets/shared/models/m2/M2Format.hpp"
#include "engine/hook/Hook.hpp"
#include "engine/hook/Registry.hpp"
#include "game/Binding.hpp"
#include "game/Gx.hpp"
#include "game/M2.hpp"
#include "offsets/engine/Gx.hpp"
#include "offsets/game/M2.hpp"

#include <cstring>

namespace
{
    namespace off = wxl::offsets::game::m2;

    using wxl::game::m2::M2SkinProfile;
    using wxl::structure::m2::M2Header;
    using wxl::structure::m2::M2SkinSection;

    off::M2_SetModelIndicesFn   g_origSetModelIndices  = nullptr;
    off::M2_SharedSetIndicesFn  g_origSharedSetIndices = nullptr;

    template <class T>
    T* At(void* base, size_t offset)
    {
        return reinterpret_cast<T*>(static_cast<uint8_t*>(base) + offset);
    }

    /**
     * @brief The submesh's first index into the skin's triangle array.
     * @param section  the submesh being placed.
     * @param skin     the skin profile it belongs to.
     * @return the widened start when level is a high half, else the client's own 16-bit value.
     */
    uint32_t TriangleStart(const M2SkinSection& section, const M2SkinProfile& skin)
    {
        const uint32_t wide = (static_cast<uint32_t>(section.level) << 16) | section.indexStart;
        // Phrased as a subtraction so a garbage level cannot wrap the bound it is being checked against.
        if (wide <= skin.indexCount && section.indexCount <= skin.indexCount - wide) return wide;
        return section.indexStart;
    }

    /** @brief True when at least one submesh of this skin needs the fold; nothing else is touched. */
    bool UsesWideStarts(const M2SkinProfile* skin)
    {
        if (!skin || !skin->submeshes || !skin->indices) return false;
        for (uint32_t i = 0; i < skin->submeshCount; ++i)
        {
            const M2SkinSection& s = skin->submeshes[i];
            if (TriangleStart(s, *skin) != s.indexStart) return true;
        }
        return false;
    }

    /**
     * @brief The client's own choice between copying the skin's indices as they stand and rebasing
     *        each submesh onto its own vertex window. Both fills below must reproduce it exactly.
     */
    bool UsesGlobalIndices(void* model)
    {
        void* record = *At<void*>(model, off::kOffSharedFileRecord);
        auto* header = *At<M2Header*>(model, off::kOffModelHeader);
        if (!record || !header) return false;
        const uint32_t flags = *At<uint32_t>(record, off::kOffM2FileFlags);
        if (flags & off::kM2FileFlagGlobalIndices) return true;
        return header->bones.count == 1 && (flags & off::kM2FileFlagSingleBoneGlobal) != 0;
    }

    /** @brief True while the buffer's contents still stand, which is when a fill is skipped. */
    bool BufferHolds(void* buffer)
    {
        return buffer && *At<uint8_t>(buffer, off::kOffGxBufBuilt) && *At<uint8_t>(buffer, off::kOffGxBufValid);
    }

    /** @brief Locks the index buffer for writing, exactly as the engine fill does. */
    void* LockBuffer(void* device, void* buffer)
    {
        return wxl::game::gx::Vtbl<off::Gx_BufLockFn>(device, static_cast<unsigned>(off::kGxVtblBufLock / sizeof(void*)))(device, buffer);
    }

    /** @brief Commits the refilled buffer and re-arms the device's index binding, as the engine does. */
    void CommitBuffer(void* device, void* buffer)
    {
        wxl::game::gx::Vtbl<off::Gx_BufUnlockFn>(device, static_cast<unsigned>(off::kGxVtblBufUnlock / sizeof(void*)))(device, buffer, 0);
        *At<uint8_t>(buffer, off::kOffGxBufBuilt) = 1;
        wxl::game::Native<off::Gx_PrimIndexPtrFn>(off::kPrimIndexPtr)(device, buffer);
    }

    /**
     * @brief Writes one submesh's block, reproducing the engine's two index spellings.
     * @param dst      cursor into the locked index buffer.
     * @param source   the skin's triangle array.
     * @param start    the submesh's first index in that array.
     * @param count    how many indices the block holds.
     * @param bias     added to every index; zero on the global-index path.
     */
    void EmitBlock(uint16_t* dst, const uint16_t* source, uint32_t start, uint32_t count, int16_t bias)
    {
        if (!bias) { std::memcpy(dst, source + start, count * sizeof(uint16_t)); return; }
        for (uint32_t i = 0; i < count; ++i)
            dst[i] = static_cast<uint16_t>(static_cast<int16_t>(source[start + i]) + bias);
    }

    /**
     * @brief Refills the per-instance index buffer: the compacted draw list, in the same order and at
     *        the same offsets the engine just used, with every block sourced from its widened start.
     */
    void RefillInstanceIndices(void* instance, const M2SkinProfile& skin, bool globalIndices)
    {
        void* device = wxl::game::gx::RawGraphicsDevice();
        void* geo = *At<void*>(instance, off::kOffInstGeometryCtx);
        if (!device || !geo) return;
        void* buffer = *At<void*>(geo, off::kOffGeoCtxIndexBuf);
        auto* groups = *At<uint8_t*>(geo, off::kOffGeoCtxGroups);
        auto* ranges = *At<uint32_t*>(geo, off::kOffGeoCtxRanges);
        auto* visible = *At<int32_t*>(instance, off::kOffInstSectionVisible);
        const uint32_t groupCount = *At<uint32_t>(geo, off::kOffGeoCtxGroupCount);
        if (!buffer || !groups || !ranges || !visible || !skin.batches) return;

        auto* dst = static_cast<uint16_t*>(LockBuffer(device, buffer));
        if (!dst) return;
        for (uint32_t g = 0; g < groupCount; ++g)
        {
            const uint32_t* range = ranges + *reinterpret_cast<uint32_t*>(groups + g * off::kGeoCtxGroupStride) * 2;
            // The vertex base restarts with each group: a group is one contiguous vertex window.
            int16_t vertexBase = 0;
            for (uint32_t b = range[0]; b <= range[1]; ++b)
            {
                const uint16_t sectionIndex = skin.batches[b].skinSectionIndex;
                if (!visible[sectionIndex]) continue;
                const M2SkinSection& s = skin.submeshes[sectionIndex];
                EmitBlock(dst, skin.indices, TriangleStart(s, skin), s.indexCount,
                          globalIndices ? int16_t(0) : static_cast<int16_t>(vertexBase - static_cast<int16_t>(s.vertexStart)));
                vertexBase = static_cast<int16_t>(vertexBase + static_cast<int16_t>(s.vertexCount));
                dst += s.indexCount;
            }
        }
        CommitBuffer(device, buffer);
    }

    /**
     * @brief Refills the shared index buffer: every submesh in skin order, each repeated once per
     *        instance copy, at the offsets the engine wrote back into the submesh copies.
     */
    void RefillSharedIndices(void* model, const M2SkinProfile& skin, bool globalIndices)
    {
        void* device = wxl::game::gx::RawGraphicsDevice();
        void* buffer = *At<void*>(model, off::kOffSharedIndexBuf);
        auto* copies = *At<M2SkinSection*>(model, off::kOffModelSubmeshBuf);
        const uint32_t instanceCopies = *At<uint32_t>(model, off::kOffSharedInstanceCopies);
        if (!device || !buffer || !copies) return;

        auto* dst = static_cast<uint16_t*>(LockBuffer(device, buffer));
        if (!dst) return;
        for (uint32_t i = 0; i < skin.submeshCount; ++i)
        {
            // Counts come from the copy, which finalize may have adjusted; the start comes from the
            // skin, which is the only place the untruncated pairing survives.
            const M2SkinSection& copy = copies[i];
            const uint32_t start = TriangleStart(skin.submeshes[i], skin);
            int16_t bias = globalIndices ? int16_t(0) : static_cast<int16_t>(-static_cast<int16_t>(copy.vertexStart));
            for (uint32_t c = 0; c < instanceCopies; ++c)
            {
                EmitBlock(dst, skin.indices, start, copy.indexCount, bias);
                dst += copy.indexCount;
                bias = static_cast<int16_t>(bias + static_cast<int16_t>(globalIndices ? skin.vertexCount : copy.vertexCount));
            }
        }
        CommitBuffer(device, buffer);
    }

    uint32_t __fastcall hkSetModelIndices(void* instance, void* edx)
    {
        void* model = instance ? *At<void*>(instance, off::kOffInstModel) : nullptr;
        auto* skin = model ? *At<M2SkinProfile*>(model, off::kOffModelSkin) : nullptr;
        void* geo = instance ? *At<void*>(instance, off::kOffInstGeometryCtx) : nullptr;
        // Read the dirty flags before the original clears them: only a real rebuild needs refilling.
        const bool rebuilding = geo && !BufferHolds(*At<void*>(geo, off::kOffGeoCtxIndexBuf));

        const uint32_t result = g_origSetModelIndices(instance, edx);
        if (!result || !rebuilding || !UsesWideStarts(skin)) return result;
        RefillInstanceIndices(instance, *skin, UsesGlobalIndices(model));
        return result;
    }

    uint32_t __fastcall hkSharedSetIndices(void* model, void* edx)
    {
        auto* skin = model ? *At<M2SkinProfile*>(model, off::kOffModelSkin) : nullptr;
        const bool rebuilding = model && !BufferHolds(*At<void*>(model, off::kOffSharedIndexBuf));

        // The original owns creating the pool/buffer pair and sizing it, so it always runs first.
        const uint32_t result = g_origSharedSetIndices(model, edx);
        if (!result || !rebuilding || !UsesWideStarts(skin)) return result;
        RefillSharedIndices(model, *skin, UsesGlobalIndices(model));
        return result;
    }

    bool InstallWideIndices()
    {
        if (!wxl::hook::Install("M2SetModelIndices", off::kSetModelIndices,
                                &hkSetModelIndices, &g_origSetModelIndices))
            return false;
        if (!wxl::hook::Install("M2SharedSetIndices", off::kSharedSetIndices,
                                &hkSharedSetIndices, &g_origSharedSetIndices))
            return false;
        WLOG_INFO("m2native-indices: submesh triangle starts read as (level << 16) | indexStart");
        return true;
    }
}

WXL_REGISTER_FEATURE("m2native-indices", true, InstallWideIndices)
