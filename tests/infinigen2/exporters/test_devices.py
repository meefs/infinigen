from types import SimpleNamespace

from infinigen2.exporters import render_cycles


def _fake_bpy():
    devices = [
        SimpleNamespace(type="OPTIX", use=True),
        SimpleNamespace(type="CUDA", use=True),
        SimpleNamespace(type="CPU", use=True),
    ]
    preferences = SimpleNamespace(
        compute_device_type="",
        devices=devices,
        get_device_types=lambda _context: [("OPTIX",), ("CUDA",)],
        get_devices_for_type=lambda _device_type: None,
    )
    context = SimpleNamespace(
        scene=SimpleNamespace(
            render=SimpleNamespace(engine="CYCLES"),
            cycles=SimpleNamespace(device="CPU"),
        ),
        preferences=SimpleNamespace(
            addons={"cycles": SimpleNamespace(preferences=preferences)}
        ),
    )
    return SimpleNamespace(context=context), preferences, devices


def test_cycles_device_environment_override(monkeypatch):
    bpy, preferences, devices = _fake_bpy()
    monkeypatch.setattr(render_cycles, "bpy", bpy)
    monkeypatch.setenv("INFINIGEN_CYCLES_DEVICE_TYPE", "CUDA")

    selected = render_cycles.configure_cycles_devices()

    assert preferences.compute_device_type == "CUDA"
    assert selected == [devices[1]]
    assert [device.use for device in devices] == [False, True, False]


def test_explicit_cycles_device_overrides_environment(monkeypatch):
    bpy, preferences, devices = _fake_bpy()
    monkeypatch.setattr(render_cycles, "bpy", bpy)
    monkeypatch.setenv("INFINIGEN_CYCLES_DEVICE_TYPE", "CUDA")

    selected = render_cycles.configure_cycles_devices("OPTIX")

    assert preferences.compute_device_type == "OPTIX"
    assert selected == [devices[0]]
    assert [device.use for device in devices] == [True, False, False]
