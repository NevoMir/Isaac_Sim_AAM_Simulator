# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0

# Warp SPH demo that uses the same base "room" and open-top container
# helpers as the PhysX ParticleSampler sample, and ties stepping to the
# Isaac Sim physics timeline (Play/Pause/Stop).

from __future__ import annotations
import math, importlib, types
from pxr import UsdGeom, Sdf, Gf, Vt

import omni.timeline
import omni.physxdemos as demo
from .ParticleDemoBaseDemo import ParticleDemoBase  # PhysX demo helpers


# ---------- small helpers (Warp kernels are generated at runtime) ----------
def _make_kernels(wp):
    @wp.func
    def _square(x: float):
        return x * x

    @wp.func
    def _cube(x: float):
        return x * x * x

    @wp.func
    def density_kernel(r: wp.vec3, h: float):
        d2 = wp.dot(r, r)
        return wp.max(_cube(_square(h) - d2), 0.0)

    @wp.func
    def diff_pressure_kernel(r: wp.vec3, p: float, pn: float, rhon: float, h: float):
        d = wp.sqrt(wp.dot(r, r))
        if d < h:
            t1 = -r / d
            t2 = (pn + p) / (2.0 * rhon)
            t3 = _square(h - d)
            return t1 * t2 * t3
        return wp.vec3()

    @wp.func
    def diff_viscous_kernel(r: wp.vec3, v: wp.vec3, vn: wp.vec3, rhon: float, h: float):
        d = wp.sqrt(wp.dot(r, r))
        if d < h:
            return (vn - v) * ((h - d) / rhon)
        return wp.vec3()

    @wp.kernel
    def compute_density(grid: wp.uint64,
                        x: wp.array(dtype=wp.vec3),
                        rho: wp.array(dtype=float),
                        kn: float, h: float):
        i = wp.hash_grid_point_id(grid, wp.tid())
        xi = x[i]
        acc = float(0.0)
        for j in wp.hash_grid_query(grid, xi, h):
            acc += density_kernel(xi - x[j], h)
        rho[i] = kn * acc

    @wp.kernel
    def get_accel(grid: wp.uint64,
                  x: wp.array(dtype=wp.vec3),
                  v: wp.array(dtype=wp.vec3),
                  rho: wp.array(dtype=float),
                  a: wp.array(dtype=wp.vec3),
                  k_iso: float, rho0: float, g: float,
                  kp: float, kv: float, h: float):
        i = wp.hash_grid_point_id(grid, wp.tid())
        xi = x[i]; vi = v[i]; rhoi = rho[i]
        pi = k_iso * (rhoi - rho0)

        fp = wp.vec3(); fv = wp.vec3()
        for j in wp.hash_grid_query(grid, xi, h):
            if j != i:
                r = x[j] - xi
                rhon = rho[j]
                pn = k_iso * (rhon - rho0)
                vn = v[j]
                fp += diff_pressure_kernel(r, pi, pn, rhon, h)
                fv += diff_viscous_kernel(r, vi, vn, rhon, h)

        f = kp * fp + kv * fv
        a[i] = f / max(rhoi, 1e-6) + wp.vec3(0.0, g, 0.0)

    @wp.kernel
    def collide_centered_open_box(x: wp.array(dtype=wp.vec3),
                                  v: wp.array(dtype=wp.vec3),
                                  damp: float,
                                  cx: float, cy: float, cz: float,
                                  W: float, H: float, L: float):
        # Axis-aligned box centered at (cx,cy,cz),
        # extents X:[cx-W/2, cx+W/2], Y:[cy, cy+H] (open top), Z:[cz-L/2, cz+L/2]
        i = wp.tid()
        xi = x[i]; vi = v[i]
        xmin = cx - 0.5 * W; xmax = cx + 0.5 * W
        ymin = cy;           ymax = cy + H
        zmin = cz - 0.5 * L; zmax = cz + 0.5 * L

        if xi[0] < xmin: xi = wp.vec3(xmin, xi[1], xi[2]); vi = wp.vec3(vi[0]*damp, vi[1], vi[2])
        if xi[0] > xmax: xi = wp.vec3(xmax, xi[1], xi[2]); vi = wp.vec3(vi[0]*damp, vi[1], vi[2])

        if xi[1] < ymin: xi = wp.vec3(xi[0], ymin, xi[2]); vi = wp.vec3(vi[0], vi[1]*damp, vi[2])
        # open top: no clamp at y > ymax

        if xi[2] < zmin: xi = wp.vec3(xi[0], xi[1], zmin); vi = wp.vec3(vi[0], vi[1], vi[2]*damp)
        if xi[2] > zmax: xi = wp.vec3(xi[0], xi[1], zmax); vi = wp.vec3(vi[0], vi[1], vi[2]*damp)

        x[i] = xi; v[i] = vi

    @wp.kernel
    def kick(v: wp.array(dtype=wp.vec3), a: wp.array(dtype=wp.vec3), dt: float):
        i = wp.tid()
        v[i] = v[i] + a[i] * dt

    @wp.kernel
    def drift(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), dt: float):
        i = wp.tid()
        x[i] = x[i] + v[i] * dt

    @wp.kernel
    def init_block(x: wp.array(dtype=wp.vec3),
                   spacing: float,
                   nx: int, ny: int, nz: int,
                   offset: wp.vec3):
        i = wp.tid()
        X = i // (ny * nz)
        Y = (i // nz) % ny
        Z = i % nz
        p = offset + spacing * wp.vec3(float(X), float(Y), float(Z))
        x[i] = p

    return types.SimpleNamespace(
        compute_density=compute_density,
        get_accel=get_accel,
        collide_centered_open_box=collide_centered_open_box,
        kick=kick, drift=drift, init_block=init_block
    )


# ------------------------------ Demo class ------------------------------
class WarpSPHDemo(demo.AsyncDemoBase):
    title = "Warp SPH Fluid (Sampler Room)"
    category = demo.Categories.PARTICLES
    short_description = "Warp SPH running in the same room & open-top container as the PhysX Particle Sampler"
    description = "Press Play to simulate. Particles fall under gravity and collide with the container walls."

    # NOTE: FloatParam requires (value, min, max, step)
    params = {
        "Smoothing_Length": demo.FloatParam(0.5, 0.1, 3.0, 0.05),
        "Packing_Ratio":    demo.FloatParam(0.55, 0.2, 1.2, 0.05),  # >1.0 => fewer particles (more spacing)
        "Width":            demo.FloatParam(70.0, 10.0, 200.0, 1.0),
        "Height":           demo.FloatParam(50.0, 10.0, 200.0, 1.0),
        "Length":           demo.FloatParam(70.0, 10.0, 200.0, 1.0),
        "Viscosity":        demo.FloatParam(0.02, 0.0, 2.0, 0.005),
        "Gravity":          demo.FloatParam(-9.81, -50.0, 0.0, 0.1),
        "Visual_Size":      demo.FloatParam(0.12, 0.02, 0.5, 0.01),
        "Use_CPU":          demo.CheckboxParam(False),
        "Show_Box":         demo.CheckboxParam(True),
    }

    kit_settings = {
        "persistent/app/viewport/displayOptions": demo.get_viewport_minimal_display_options_int(),
        "rtx/translucency/maxRefractionBounces": 8,
    }

    def __init__(self):
        # AsyncDemoBase wires up timeline callbacks once _setup_callbacks() is called in create()
        super().__init__(enable_fabric=False, fabric_compatible=False)

        # Borrow the PhysX demo helpers to build the same room & box
        ParticleDemoBase.__init__(self, enable_fabric=False, fabric_compatible=False)

        self._stage = None
        self._root = None
        self._points = None

        # Warp + kernels
        self.wp = None
        self.k = None
        self.device = None

        # sim parameters (defaults; overwritten in create)
        self.h = 0.5
        self.pack = 0.55
        self.W = self.H = self.L = 70.0
        self.nu = 0.02
        self.g = -9.81
        self.visual_size = 0.12
        self.damp = -0.95

        # derived
        self.substeps = 12
        self.kn = self.kp = self.kv = 0.0

        # data
        self.n = 0
        self.x = self.v = self.rho = self.a = None
        self.grid = None

        # container center (matches PhysX helper translation)
        self._box_center = Gf.Vec3f(0.0, -5.0, 0.0)

        # run state (handled by AsyncDemoBase via on_timeline_event)
        self._is_running = False

    # ---------- utilities ----------
    def _to_v3(self, arr):
        out = Vt.Vec3fArray(len(arr))
        for i, p in enumerate(arr):
            out[i] = Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
        return out

    # ---------- lifecycle ----------
    def create(self, stage, Smoothing_Length, Packing_Ratio, Width, Height, Length,
               Viscosity, Gravity, Visual_Size, Use_CPU, Show_Box):
        self._stage = stage
        self._setup_callbacks()  # important: wire timeline -> on_timeline_event/on_physics_step

        # Build the same base scene (room, lighting, physicsScene)
        default_prim_path = ParticleDemoBase.setup_base_scene(self, stage)
        self._root = default_prim_path.AppendChild("WarpSPH")
        UsdGeom.Xform.Define(stage, self._root)

        # Optional: same open-top container from the PhysX helper
        if Show_Box:
            ParticleDemoBase.create_particle_box_collider(
                self,
                default_prim_path.AppendChild("box"),
                side_length=float(Width),
                height=float(Height),
                thickness=4.0,
                translate=Gf.Vec3f(self._box_center[0], self._box_center[1], self._box_center[2]),
                add_cylinder_top=False,
            )

        # Parameters
        self.h = float(Smoothing_Length)
        self.pack = float(Packing_Ratio)      # spacing multiplier (1.0 = tight, >1 = fewer particles)
        self.W = float(Width)
        self.H = float(Height)
        self.L = float(Length)
        self.nu = float(Viscosity)
        self.g = float(Gravity)
        self.visual_size = float(Visual_Size)
        self.substeps = max(1, int(24.0 / self.h))
        self.device = "cpu" if Use_CPU else None

        # Warp + kernels
        self.wp = importlib.import_module("warp")
        self.k = _make_kernels(self.wp)

        # Kernel constants (same as original example)
        m = 0.01 * self.h**3
        self.kn = (315.0 * m) / (64.0 * math.pi * self.h**9)
        self.kp = -(45.0 * m) / (math.pi * self.h**6)
        self.kv = (45.0 * self.nu * m) / (math.pi * self.h**6)

        # Particle block (fewer particles by increasing spacing with pack)
        spacing = max(0.15 * self.h, self.pack * self.h)
        nx = max(1, int((0.40 * self.W) / spacing))
        ny = max(1, int((0.50 * self.H) / spacing))
        nz = max(1, int((0.40 * self.L) / spacing))
        self.n = nx * ny * nz

        # Place the block above the box center and a bit to the (-x, +y) as requested
        block_offset = self.wp.vec3(
            float(self._box_center[0] - 0.20 * self.W),
            float(self._box_center[1] + 0.75 * self.H),
            float(self._box_center[2]),
        )

        wp = self.wp
        with wp.ScopedDevice(self.device):
            self.x = wp.empty(self.n, dtype=wp.vec3)
            self.v = wp.zeros(self.n, dtype=wp.vec3)
            self.rho = wp.zeros(self.n, dtype=float)
            self.a = wp.zeros(self.n, dtype=wp.vec3)

            wp.launch(self.k.init_block, dim=self.n,
                      inputs=[self.x, spacing, nx, ny, nz, block_offset], device=self.device)

            # Hash grid size heuristic
            gs = max(1, int(self.H / (2.0 * self.h)))
            self.grid = wp.HashGrid(gs, gs, gs)

        # Visual points
        pts_path = self._root.AppendPath("Particles")
        self._points = UsdGeom.Points.Define(stage, pts_path)
        self._points.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.9, 0.7, 0.6)]))
        diameter = max(0.02, self.visual_size * self.h)
        self._points.CreateWidthsAttr().Set(Vt.FloatArray([diameter] * self.n))
        self._points.CreatePointsAttr().Set(self._to_v3(self.x.numpy()))

    # AsyncDemoBase sends us these:
    def on_timeline_event(self, e):
        TT = omni.timeline.TimelineEventType
        if e.type == TT.PLAY:
            self._is_running = True
        elif e.type in (TT.PAUSE, TT.STOP):
            self._is_running = False

    def on_physics_step(self, dt):
        if not self._is_running or self.x is None:
            return
        self._step_sph(dt if dt and dt > 0.0 else (1.0 / 60.0))

    def update(self, stage, dt, viewport, physxIFace):
        # push positions to USD every frame
        if self._points and self.x is not None:
            self._points.GetPointsAttr().Set(self._to_v3(self.x.numpy()))

    def on_shutdown(self):
        self._points = None
        self.x = self.v = self.rho = self.a = self.grid = None
        ParticleDemoBase.on_shutdown(self)

    # ---------- SPH step ----------
    def _step_sph(self, dt):
        wp = self.wp; k = self.k
        # a few micro-steps per frame
        sub_dt = 0.008
        steps = max(1, int(max(dt, 1.0 / 60.0) / sub_dt))
        with wp.ScopedDevice(self.device):
            for _ in range(steps):
                self.grid.build(self.x, self.h, device=self.device)

                wp.launch(k.compute_density, dim=self.n,
                          inputs=[self.grid.id, self.x, self.rho, self.kn, self.h],
                          device=self.device)

                wp.launch(k.get_accel, dim=self.n,
                          inputs=[self.grid.id, self.x, self.v, self.rho, self.a,
                                  20.0, 1.0, self.g, self.kp, self.kv, self.h],
                          device=self.device)

                wp.launch(k.collide_centered_open_box, dim=self.n,
                          inputs=[self.x, self.v, self.damp,
                                  float(self._box_center[0]), float(self._box_center[1]), float(self._box_center[2]),
                                  self.W, self.H, self.L],
                          device=self.device)

                wp.launch(k.kick, dim=self.n, inputs=[self.v, self.a, sub_dt], device=self.device)
                wp.launch(k.drift, dim=self.n, inputs=[self.x, self.v, sub_dt], device=self.device)