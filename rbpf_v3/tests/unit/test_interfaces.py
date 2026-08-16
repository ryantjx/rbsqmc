import inspect

from rbpf_v3.src import smoothing, smoothing_noncuthbert


def test_backends_have_matching_public_contracts():
    assert smoothing.MCEMConfig._fields == smoothing_noncuthbert.MCEMConfig._fields
    assert smoothing.BackwardDiagnostics._fields == smoothing_noncuthbert.BackwardDiagnostics._fields
    assert smoothing.SmoothedStates._fields == smoothing_noncuthbert.SmoothedStates._fields
    for name in (
        "E_step",
        "run_mcem",
        "rb_backward_simulation",
        "batched_backward_step",
        "complete_data_terms",
    ):
        assert tuple(inspect.signature(getattr(smoothing, name)).parameters) == tuple(
            inspect.signature(getattr(smoothing_noncuthbert, name)).parameters
        )


def test_no_shared_smoothing_module_or_cross_imports():
    source_c = inspect.getsource(smoothing)
    source_d = inspect.getsource(smoothing_noncuthbert)
    assert "smoothing_common" not in source_c + source_d
    assert "smoothing_noncuthbert" not in source_c
    assert "src.smoothing import" not in source_d
