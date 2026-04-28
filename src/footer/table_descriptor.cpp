// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/footer/table_descriptor.cpp
// ────────────────────────────────────────────────────────
#include "fls/footer/table_descriptor.hpp"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/verifier.h" // flatbuffers::Verifier
#include "fls/common/alias.hpp"
#include "fls/footer/rowgroup_descriptor.hpp"
#include "fls/footer/table_descriptor_generated.h"
#include "fls/io/file.hpp"
#include "fls/std/filesystem.hpp"
#include "fls/std/vector.hpp"
#include "fls/table/table.hpp"
#include <cstddef> // std::size_t
#include <cstdint> // uint8_t
#include <cstring> // std::memcpy
#include <memory> // std::shared_ptr, std::make_shared
#include <stdexcept>
#include <utility> // std::move
#include <vector>

namespace fastlanes {

const TableDescriptor& get_table_descriptor(const uint8_t* data, std::size_t size) {
	if (!data || size == 0) {
		throw std::invalid_argument("get_table_descriptor: null data or zero size");
	}
#if defined(DEBUG)
	{
		flatbuffers::Verifier v(data, size);
		if (!VerifyTableDescriptorBuffer(v)) {
			throw std::runtime_error("get_table_descriptor: invalid TableDescriptor FlatBuffer");
		}
	}
#endif
	return *GetTableDescriptor(data);
}

namespace {

vector<uint8_t> read_file_bytes(File& file) {
	const auto size = static_cast<std::size_t>(file.Size());
	if (size == 0) {
		throw std::runtime_error("TableDescriptorHandle::FromFile: empty file");
	}

	vector<uint8_t> storage(size);
	file.ReadRange(storage.data(), 0, size);
	return storage;
}

vector<uint8_t> read_file_slice(File& file, n_t offset, n_t size) {
	const auto file_size = file.Size();
	if (offset > file_size || size > file_size - offset) {
		throw std::runtime_error("TableDescriptorHandle::FromFileSlice: read range exceeds file size");
	}

	vector<uint8_t> storage(static_cast<std::size_t>(size));
	if (size > 0) {
		file.ReadRange(storage.data(), offset, size);
	}
	return storage;
}

} // namespace

const TableDescriptor& make_table_descriptor(const path& file_path, std::vector<uint8_t>& storage) {
	File file(file_path);
	storage = read_file_bytes(file);
	return get_table_descriptor(storage.data(), storage.size());
}

const TableDescriptor&
make_table_descriptor(const path& file_path, n_t offset, n_t size, std::vector<uint8_t>& storage) {
	File file(file_path);
	storage = read_file_slice(file, offset, size);
	return get_table_descriptor(storage.data(), storage.size());
}

const TableDescriptor* TableDescriptorHandle::MakePtr(const std::shared_ptr<vector<uint8_t>>& bytes, bool verify) {
	if (!bytes || bytes->empty()) {
		throw std::runtime_error("TableDescriptorHandle: empty buffer");
	}

	const uint8_t*    data = bytes->data();
	const std::size_t sz   = bytes->size();

#if defined(DEBUG)
	(void)verify; // always verify in DEBUG
	flatbuffers::Verifier v(data, sz);
	if (!VerifyTableDescriptorBuffer(v)) {
		throw std::runtime_error("TableDescriptorHandle: verification failed (DEBUG)");
	}
#else
	if (verify) {
		flatbuffers::Verifier v(data, sz);
		if (!VerifyTableDescriptorBuffer(v)) {
			throw std::runtime_error("TableDescriptorHandle: verification failed");
		}
	}
#endif

	return GetTableDescriptor(data);
}

TableDescriptorHandle TableDescriptorHandle::FromBytes(vector<uint8_t> bytes, bool verify) {
	auto sp  = std::make_shared<vector<uint8_t>>(std::move(bytes));
	auto ptr = MakePtr(sp, verify);
	return {std::move(sp), ptr};
}

TableDescriptorHandle TableDescriptorHandle::FromFile(const path& file_path, bool verify) {
	File file(file_path);
	return FromFile(file, verify);
}

TableDescriptorHandle TableDescriptorHandle::FromFile(File& file, bool verify) {
	return FromBytes(read_file_bytes(file), verify);
}

TableDescriptorHandle TableDescriptorHandle::FromFileSlice(const path& file_path, n_t offset, n_t size, bool verify) {
	File file(file_path);
	return FromFileSlice(file, offset, size, verify);
}

TableDescriptorHandle TableDescriptorHandle::FromFileSlice(File& file, n_t offset, n_t size, bool verify) {
	return FromBytes(read_file_slice(file, offset, size), verify);
}

TableDescriptorHandle TableDescriptorHandle::FromNative(const TableDescriptorT& native) {
	flatbuffers::FlatBufferBuilder fbb;
	auto                           off = TableDescriptor::Pack(fbb, &native);
	fbb.Finish(off);

	auto            det = fbb.Release();
	vector<uint8_t> bytes(det.size());
	if (det.size() > 0) {
		std::memcpy(bytes.data(), det.data(), det.size());
	}

	return FromBytes(std::move(bytes), /*verify=*/false);
}

up<TableDescriptorT> make_table_descriptor(const Table& table) {
	auto table_descriptor = make_unique<TableDescriptorT>();

	for (n_t rowgroup_idx = 0; rowgroup_idx < table.get_n_rowgroups(); ++rowgroup_idx) {
		table_descriptor->m_rowgroup_descriptors.push_back(make_rowgroup_descriptor(*table.m_rowgroups[rowgroup_idx]));
	}
	table_descriptor->m_table_binary_size = 0;

	return table_descriptor;
}

} // namespace fastlanes
