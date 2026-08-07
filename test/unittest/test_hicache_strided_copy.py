"""Unit tests for unified-memory HiCache strided copy optimization.

These tests verify that ``_to_2d_view`` correctly handles both contiguous
and strided tensors, and that the data integrity is preserved when copying
through the optimized path (no staging buffer).

Tests are designed to run on CPU only (no GPU required).
"""

import unittest

import torch


def _to_2d_view(tensor: torch.Tensor, element_dim: int) -> torch.Tensor:
    """Reshape a (possibly strided) KV-cache tensor to 2-D (-1, element_dim).

    This is a copy of the function in sglang.kernels.ops.kvcache.hicache,
    inlined here to avoid importing the full sglang package on Windows
    (which requires the Unix-only ``resource`` module).
    """
    # Fast path: .view() works for contiguous or compatible-stride tensors.
    try:
        return tensor.view(-1, element_dim)
    except RuntimeError:
        pass
    # Fallback: build a 2-D strided view manually.
    num_slots = tensor.numel() // element_dim
    stride0 = tensor.stride(0) if tensor.dim() > 0 else element_dim
    if tensor.dim() >= 2:
        stride0 = tensor.stride(0)
        num_slots = tensor.size(0) * tensor.size(1) if tensor.size(1) > 1 else tensor.size(0)
        if tensor.size(1) == 1:
            num_slots = tensor.size(0)
    return torch.as_strided(tensor, size=(num_slots, element_dim), stride=(stride0, 1))


class TestTo2dView(unittest.TestCase):
    """Test the _to_2d_view helper that reshapes KV tensors to 2-D."""

    def test_contiguous_tensor(self):
        """Contiguous tensors should produce a correct 2-D view."""
        num_slots = 16
        head_num = 8
        head_dim = 128
        element_dim = head_num * head_dim  # 1024

        tensor = torch.randn(num_slots, head_num, head_dim)
        result = _to_2d_view(tensor, element_dim)

        self.assertEqual(result.shape, (num_slots, element_dim))
        self.assertTrue(torch.equal(result, tensor.view(-1, element_dim)))

    def test_strided_tensor_page_size_1(self):
        """Strided tensor (page_size=1, envelope-major) should produce correct 2-D view.

        Simulates the unified-memory layout:
          shape = (num_pages, 1, head_num, head_dim)
          stride = (stride_page, element_dim, head_dim, 1)
        """
        num_pages = 16
        head_num = 8
        head_dim = 128
        element_dim = head_num * head_dim  # 1024
        layer_num = 4
        k_row_bytes = head_num * head_dim
        v_row_bytes = head_num * head_dim
        entry_bytes = layer_num * (k_row_bytes + v_row_bytes)
        stride_page = entry_bytes  # in elements

        # Create a raw buffer large enough
        raw = torch.randn(num_pages * entry_bytes)

        k_shape = (num_pages, 1, head_num, head_dim)
        k_stride = (stride_page, element_dim, head_dim, 1)

        # Create strided view for layer 0's K block
        k_base = 0  # layer 0
        k_view = torch.as_strided(
            raw, size=k_shape, stride=k_stride, storage_offset=k_base
        )

        result = _to_2d_view(k_view, element_dim)

        self.assertEqual(result.shape, (num_pages, element_dim))
        # Verify data integrity: result[i] should equal k_view[i, 0].flatten()
        for i in range(num_pages):
            expected = k_view[i, 0].flatten()
            actual = result[i]
            self.assertTrue(torch.equal(actual, expected), f"Mismatch at slot {i}")

    def test_strided_tensor_preserves_data(self):
        """Data read through _to_2d_view should match direct indexing."""
        num_pages = 8
        head_num = 4
        head_dim = 64
        element_dim = head_num * head_dim  # 256
        layer_num = 3
        k_row_bytes = head_num * head_dim
        v_row_bytes = head_num * head_dim
        entry_bytes = layer_num * (k_row_bytes + v_row_bytes)
        stride_page = entry_bytes

        raw = torch.randn(num_pages * entry_bytes)

        k_shape = (num_pages, 1, head_num, head_dim)
        k_stride = (stride_page, element_dim, head_dim, 1)

        # Layer 1's K block
        k_base = 1 * (k_row_bytes + v_row_bytes)
        k_view = torch.as_strided(
            raw, size=k_shape, stride=k_stride, storage_offset=k_base
        )

        result = _to_2d_view(k_view, element_dim)

        # Verify each slot
        for i in range(num_pages):
            expected = k_view[i, 0].flatten()
            self.assertTrue(torch.equal(result[i], expected))

    def test_strided_tensor_stride_correct(self):
        """The stride of the 2-D view should match the original tensor's page stride."""
        num_pages = 4
        head_num = 4
        head_dim = 32
        element_dim = head_num * head_dim  # 128
        layer_num = 2
        k_row_bytes = head_num * head_dim
        v_row_bytes = head_num * head_dim
        entry_bytes = layer_num * (k_row_bytes + v_row_bytes)
        stride_page = entry_bytes

        raw = torch.randn(num_pages * entry_bytes)

        k_shape = (num_pages, 1, head_num, head_dim)
        k_stride = (stride_page, element_dim, head_dim, 1)

        k_view = torch.as_strided(raw, size=k_shape, stride=k_stride)

        result = _to_2d_view(k_view, element_dim)

        # stride[0] should be stride_page (the per-slot stride)
        self.assertEqual(result.stride(0), stride_page)
        self.assertEqual(result.stride(1), 1)



class TestStridedCopyDataIntegrity(unittest.TestCase):
    """Test that copying data through strided views preserves data integrity.

    These tests simulate what the JIT kernel does: read from a strided source
    and write to a contiguous destination (and vice versa), using PyTorch
    operations that mirror the kernel's pointer-based access pattern.
    """

    def test_d2h_strided_to_contiguous(self):
        """Simulate D2H: read from strided GPU buffer, write to contiguous host buffer.

        Uses index_select to simulate the kernel's gather operation.
        """
        num_slots = 16
        head_num = 8
        head_dim = 128
        element_dim = head_num * head_dim  # 1024
        layer_num = 4
        k_row_bytes = head_num * head_dim
        v_row_bytes = head_num * head_dim
        entry_bytes = layer_num * (k_row_bytes + v_row_bytes)
        stride_page = entry_bytes

        # Create raw buffer (simulating unified memory)
        raw = torch.randn(num_slots * entry_bytes)

        # Create strided K view for layer 1
        k_shape = (num_slots, 1, head_num, head_dim)
        k_stride = (stride_page, element_dim, head_dim, 1)
        k_base = 1 * (k_row_bytes + v_row_bytes)
        k_view = torch.as_strided(
            raw, size=k_shape, stride=k_stride, storage_offset=k_base
        )

        # Flatten to 2D
        k_2d = k_view.view(num_slots, element_dim)

        # Simulate D2H: gather specific slots
        indices = torch.tensor([0, 3, 7, 12, 15])
        gathered = k_2d[indices]  # This is what the kernel does

        # Create contiguous host buffer
        host_buf = torch.empty(len(indices), element_dim)
        host_buf[:] = gathered

        # Verify
        for i, idx in enumerate(indices):
            expected = k_view[idx, 0].flatten()
            self.assertTrue(torch.equal(host_buf[i], expected))

    def test_h2d_contiguous_to_strided(self):
        """Simulate H2D: read from contiguous host buffer, write to strided GPU buffer.

        Uses index_copy_ to simulate the kernel's scatter operation.
        """
        num_slots = 16
        head_num = 8
        head_dim = 128
        element_dim = head_num * head_dim  # 1024
        layer_num = 4
        k_row_bytes = head_num * head_dim
        v_row_bytes = head_num * head_dim
        entry_bytes = layer_num * (k_row_bytes + v_row_bytes)
        stride_page = entry_bytes

        # Create raw buffer (simulating unified memory)
        raw = torch.zeros(num_slots * entry_bytes)

        # Create strided K view for layer 1
        k_shape = (num_slots, 1, head_num, head_dim)
        k_stride = (stride_page, element_dim, head_dim, 1)
        k_base = 1 * (k_row_bytes + v_row_bytes)
        k_view = torch.as_strided(
            raw, size=k_shape, stride=k_stride, storage_offset=k_base
        )

        # Create contiguous host data
        indices = torch.tensor([0, 3, 7, 12, 15])
        host_data = torch.randn(len(indices), element_dim)

        # Simulate H2D: scatter to strided buffer
        k_2d = k_view.view(num_slots, element_dim)
        k_2d[indices] = host_data

        # Verify
        for i, idx in enumerate(indices):
            expected = host_data[i].view(head_num, head_dim)
            self.assertTrue(torch.equal(k_view[idx, 0], expected))

    def test_round_trip_strided(self):
        """Round-trip: write to strided buffer, read back, verify data integrity."""
        num_slots = 32
        head_num = 4
        head_dim = 64
        element_dim = head_num * head_dim  # 256
        layer_num = 3
        k_row_bytes = head_num * head_dim
        v_row_bytes = head_num * head_dim
        entry_bytes = layer_num * (k_row_bytes + v_row_bytes)
        stride_page = entry_bytes

        raw = torch.zeros(num_slots * entry_bytes)

        k_shape = (num_slots, 1, head_num, head_dim)
        k_stride = (stride_page, element_dim, head_dim, 1)
        k_base = 2 * (k_row_bytes + v_row_bytes)
        k_view = torch.as_strided(
            raw, size=k_shape, stride=k_stride, storage_offset=k_base
        )

        # Write data
        original_data = torch.randn(num_slots, element_dim)
        k_2d = k_view.view(num_slots, element_dim)
        k_2d[:] = original_data

        # Read back
        read_back = k_2d.clone()

        # Verify
        self.assertTrue(torch.equal(original_data, read_back))

    def test_multi_layer_independence(self):
        """Verify that writing to one layer's strided view doesn't corrupt another layer."""
        num_slots = 8
        head_num = 4
        head_dim = 32
        element_dim = head_num * head_dim  # 128
        layer_num = 3
        k_row_bytes = head_num * head_dim
        v_row_bytes = head_num * head_dim
        entry_bytes = layer_num * (k_row_bytes + v_row_bytes)
        stride_page = entry_bytes

        raw = torch.zeros(num_slots * entry_bytes)

        k_shape = (num_slots, 1, head_num, head_dim)
        k_stride = (stride_page, element_dim, head_dim, 1)

        # Write to layer 0
        k_view_0 = torch.as_strided(
            raw, size=k_shape, stride=k_stride, storage_offset=0
        )
        data_0 = torch.randn(num_slots, element_dim)
        k_view_0.view(num_slots, element_dim)[:] = data_0

        # Write to layer 1
        k_base_1 = 1 * (k_row_bytes + v_row_bytes)
        k_view_1 = torch.as_strided(
            raw, size=k_shape, stride=k_stride, storage_offset=k_base_1
        )
        data_1 = torch.randn(num_slots, element_dim)
        k_view_1.view(num_slots, element_dim)[:] = data_1

        # Verify layer 0 is unchanged
        read_0 = k_view_0.view(num_slots, element_dim).clone()
        self.assertTrue(torch.equal(data_0, read_0))

        # Verify layer 1 is correct
        read_1 = k_view_1.view(num_slots, element_dim).clone()
        self.assertTrue(torch.equal(data_1, read_1))


if __name__ == "__main__":
    unittest.main()
