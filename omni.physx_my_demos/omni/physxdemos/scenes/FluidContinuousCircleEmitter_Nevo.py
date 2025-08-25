import math
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, Vt, UsdPhysics, PhysxSchema
from omni.physx.scripts import utils
from omni.physx.scripts import physicsUtils, particleUtils
import omni.physxdemos as demo
import omni.physx.bindings._physx as physx_settings_bindings
import omni.timeline
import omni.usd
import numpy as np


class FluidBallEmitterDemo(demo.AsyncDemoBase):
    title = "Continuous Circle Fluid Emitter (Demo Scene, Props Removed) — Half Size"
    category = demo.Categories.PARTICLES
    short_description = "PBD fluid in the premade scene; particle size drives physics"
    description = "Emits PBD fluid along a circular path; particles under /World; size controls physics & visuals."

    params = {
        "Single_Particle_Set": demo.CheckboxParam(True),
        "Use_Instancer": demo.CheckboxParam(True),
    }

    kit_settings = {
        "persistent/app/viewport/displayOptions": demo.get_viewport_minimal_display_options_int(),
        physx_settings_bindings.SETTING_MIN_FRAME_RATE: 60,
        physx_settings_bindings.SETTING_UPDATE_VELOCITIES_TO_USD: True,
        "rtx/post/aa/op": 3,
        "rtx/post/dlss/execMode": 0,
    }

    def __init__(self):
        super().__init__(enable_fabric=False, fabric_compatible=False)
        # timeline state
        self._time = 0.0
        self._is_running = False
        self._is_paused = False

        # rng
        self._rng_seed = 42
        self._rng = np.random.default_rng(self._rng_seed)

        # ---------- SIZE KNOBS ----------
        self._particle_size = 0.003  # <<< RADIUS in meters (physics + visuals)
        self._emit_radius   = 0.007  # <<< RADIUS in meters (nozzle opening)

        # emission kinematics
        self._emit_enabled = True
        self._emit_rate_particles_per_second = 3000
        self._emit_velocity = Gf.Vec3f(0.0, 0.0, -20.0)
        self._vel_jitter = 0.0
        self._pos_jitter = 0.0
        self._emit_accum = 0.0

        # capacity
        self._max_particles = 100_000

        # visuals / colors
        self._num_colors = 20
        self._color_cycle = True

        # anisotropy controls (helps reduce big newborn look)
        self._useAnisotropy = False
        self._anisotropy_scale = 0.7

        # USD / stage handles
        self._isActive = True
        self._stage = None
        self._sharedParticlePrim = None
        self._session_sub_layer = None

        self._fluid_rest_offset = None

        # ORBITING NOZZLE — /World coords
        self._orbit_enabled = True
        self._orbit_center = Gf.Vec3f(0.0, 0.0, 0.7)  # 1 m high
        self._orbit_radius = 0.1                      # 0.2 m diameter circle
        self._orbit_omega = 2.0 * math.pi * 0.2       # 0.2 rev/s
        self._orbit_phase = 0.0
        self._nozzle_pos = self._orbit_center + Gf.Vec3f(self._orbit_radius, 0.0, 0.0)

        # Debug gizmo for emitter position
        self._debug_nozzle_viz = True
        self._nozzle_viz_prim = None

    # ---------- utility: colors ----------
    def create_colors(self):
        fractions = np.linspace(0.0, 1.0, self._num_colors)
        return [self.create_color(frac) for frac in fractions]

    def create_color(self, frac):
        hue = frac
        saturation = 1.0
        luminosity = 0.5
        hue6 = hue * 6.0
        modulo = Gf.Vec3f((hue6 + 0.0) % 6.0, (hue6 + 4.0) % 6.0, (hue6 + 2.0) % 6.0)
        absolute = Gf.Vec3f(abs(modulo[0] - 3.0), abs(modulo[1] - 3.0), abs(modulo[2] - 3.0))
        rgb = Gf.Vec3f(
            Gf.Clampf(absolute[0] - 1.0, 0.0, 1.0),
            Gf.Clampf(absolute[1] - 1.0, 0.0, 1.0),
            Gf.Clampf(absolute[2] - 1.0, 0.0, 1.0),
        )
        linter = Gf.Vec3f(1.0) * (1.0 - saturation) + rgb * saturation
        rgb = 0.5 * linter
        return rgb

    # ---------- USD helpers ----------
    @staticmethod
    def extend_array_attribute(attribute, elements):
        curr = attribute.Get()
        if curr is not None:
            arr = list(curr)
            arr.extend(elements)
            attribute.Set(arr)
        else:
            attribute.Set(elements)

    def add_shared_particles(self, positions_list, velocities_list, color_index):
        particleSet = PhysxSchema.PhysxParticleSetAPI(self._sharedParticlePrim)
        pointInstancer = UsdGeom.PointInstancer(self._sharedParticlePrim)
        points = UsdGeom.Points(self._sharedParticlePrim)

        # write sim positions first if smoothing enabled (avoids one-frame mismatch)
        simPointsAttr = particleSet.GetSimulationPointsAttr()
        if not simPointsAttr.HasAuthoredValue():
            simPointsAttr.Set(Vt.Vec3fArray([]))
        self.extend_array_attribute(simPointsAttr, positions_list)

        if pointInstancer:
            self.extend_array_attribute(pointInstancer.GetPositionsAttr(), positions_list)
            self.extend_array_attribute(pointInstancer.GetVelocitiesAttr(), velocities_list)
            self.extend_array_attribute(pointInstancer.GetProtoIndicesAttr(), [color_index] * len(positions_list))
            self.extend_array_attribute(pointInstancer.GetOrientationsAttr(), [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * len(positions_list))
            if not pointInstancer.GetScalesAttr().HasAuthoredValue():
                pointInstancer.GetScalesAttr().Set(Vt.Vec3fArray([]))
        elif points:
            self.extend_array_attribute(points.GetPointsAttr(), positions_list)
            self.extend_array_attribute(points.GetVelocitiesAttr(), velocities_list)
            self.extend_array_attribute(points.GetWidthsAttr(), [2 * self._particle_size] * len(positions_list))
            primVars = points.GetDisplayColorPrimvar()
            primVarsIndicesAttr = primVars.GetIndicesAttr()
            self.extend_array_attribute(primVarsIndicesAttr, [color_index] * len(positions_list))

    # ---------- scene cleaning ----------
    def _remove_center_props(self, stage):
        world = stage.GetPrimAtPath("/World")
        if not world:
            return
        to_delete = set()
        candidates = [
            "/World/Room/Props", "/World/room/Props", "/World/Props",
            "/World/Table", "/World/table",
            "/World/Room/Table", "/World/room/Table",
        ]
        for p in candidates:
            prim = stage.GetPrimAtPath(p)
            if prim and prim.IsValid():
                to_delete.add(prim.GetPath())

        for prim in Usd.PrimRange(world):
            if prim == world:
                continue
            name = prim.GetName().lower()
            path_str = prim.GetPath().pathString

            if prim.IsA(UsdGeom.Camera):
                continue
            if prim.IsA(UsdLux.DistantLight) or prim.IsA(UsdLux.RectLight) or prim.IsA(UsdLux.SphereLight) or prim.IsA(UsdLux.DomeLight):
                continue
            if path_str.endswith("/physicsScene"):
                continue
            if "ground" in name or "floor" in name:
                continue

            if "table" in name or "desk" in name or name == "props":
                to_delete.add(prim.GetPath())

        for p in sorted(to_delete, key=lambda s: len(str(s).split("/")), reverse=True):
            try:
                stage.RemovePrim(p)
            except Exception:
                pass

    # remove windows/walls (visuals) AND their colliders; keep floor + floor collider
    def _remove_walls_windows_and_props(self, stage):
        paths = [
            "/World/roomScene/walls",
            "/World/roomScene/windows",
            "/World/Room/walls", "/World/Room/windows",
            "/World/room/walls", "/World/room/windows",
            "/World/roomScene/colliders/walls",
            "/World/roomScene/colliders/windows",
            "/World/Room/colliders/walls",
            "/World/Room/colliders/windows",
            "/World/room/colliders/walls",
            "/World/room/colliders/windows",
        ]
        to_delete = set()
        for p in paths:
            prim = stage.GetPrimAtPath(p)
            if prim and prim.IsValid():
                to_delete.add(prim.GetPath())

        room = stage.GetPrimAtPath("/World/roomScene")
        if room and room.IsValid():
            for prim in Usd.PrimRange(room):
                name = prim.GetName().lower()
                path = prim.GetPath().pathString.lower()
                if "floor" in path:
                    continue
                if ("wall" in name or "window" in name) and "colliders/floor" not in path:
                    to_delete.add(prim.GetPath())

        self._remove_center_props(stage)

        for p in sorted(to_delete, key=lambda s: len(str(s).split("/")), reverse=True):
            try:
                stage.RemovePrim(p)
            except Exception:
                pass

    # ---------------- helpers: transform & scale Z on floors ----------------
    def _set_translate_z(self, stage, path, z):
        prim = stage.GetPrimAtPath(path)
        if not (prim and prim.IsValid()):
            return
        xf = UsdGeom.Xformable(prim)
        t_op = None
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                t_op = op
                break
        if t_op is None:
            t_op = xf.AddTranslateOp()
        curr = t_op.Get() or Gf.Vec3f(0.0, 0.0, 0.0)
        t_op.Set(Gf.Vec3f(curr[0], curr[1], float(z)))

    def _set_scale_z(self, stage, path, z_scale):
        prim = stage.GetPrimAtPath(path)
        if not (prim and prim.IsValid()):
            return
        xf = UsdGeom.Xformable(prim)
        s_op = None
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                s_op = op
                break
        if s_op is None:
            s_op = xf.AddScaleOp()
        curr = s_op.Get() or Gf.Vec3f(1.0, 1.0, 1.0)
        s_op.Set(Gf.Vec3f(curr[0], curr[1], float(z_scale)))

    def _thicken_and_reposition_floors(self, stage):
        self._set_translate_z(stage, "/World/roomScene/colliders/floor/infinitePlane", -1.0)
        self._set_scale_z(    stage, "/World/roomScene/colliders/floor/infinitePlane",  1.0)

        self._set_translate_z(stage, "/World/roomScene/colliders/floor/mainFloorActor", -0.9)
        self._set_scale_z(    stage, "/World/roomScene/colliders/floor/mainFloorActor",  1.0)

        self._set_translate_z(stage, "/World/roomScene/renderables/groundPlane0", -0.4)
        self._set_scale_z(    stage, "/World/roomScene/renderables/groundPlane0",  1.0)

        self._set_translate_z(stage, "/World/roomScene/renderables/groundPlane1", 0.5)
        self._set_scale_z(    stage, "/World/roomScene/renderables/groundPlane1",  1.0)
    # -----------------------------------------------------------------------

    # ---------- physics sizing ----------
    def _compute_particle_system_offsets(self, r):
        fluid_rest_offset = float(r)
        rest_offset = fluid_rest_offset / 0.6
        solid_rest_offset = rest_offset
        particle_contact_offset = max(solid_rest_offset + 0.25 * r, rest_offset)
        contact_offset = rest_offset + 0.25 * r
        return contact_offset, rest_offset, particle_contact_offset, solid_rest_offset, fluid_rest_offset

    # ---------- nozzle viz ----------
    def _ensure_nozzle_viz(self, stage):
        if not self._debug_nozzle_viz or self._nozzle_viz_prim:
            return
        path = Sdf.Path("/World/Nozzle")
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr(0.02)
        sphere.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.2, 0.2)]))
        self._nozzle_viz_prim = sphere.GetPrim()

    def _update_nozzle_viz(self):
        if not (self._debug_nozzle_viz and self._nozzle_viz_prim):
            return
        xf = UsdGeom.Xformable(self._nozzle_viz_prim)
        if not xf.GetOrderedXformOps():
            xf.AddTranslateOp()
        xf.SetXformOpOrder([op.GetOpName() for op in xf.GetOrderedXformOps()])
        xf.GetOrderedXformOps()[0].Set(Gf.Vec3f(self._nozzle_pos[0], self._nozzle_pos[1], self._nozzle_pos[2]))

    # ---------- scene setup ----------
    def create(self, stage, Single_Particle_Set, Use_Instancer):
        self._stage = stage
        self._setup_callbacks()

        self._useSharedParticleSet = True
        self._usePointInstancer = True
        self._useSmoothing = True

        defaultPrimPath, scene = demo.setup_physics_scene(self, stage, metersPerUnit=1.0)
        scenePath = defaultPrimPath + "/physicsScene"

        demo.get_demo_room(self, stage)
        self._remove_walls_windows_and_props(stage)
        self._thicken_and_reposition_floors(stage)

        self._particleSystemPath = Sdf.Path("/World/particleSystem0")

        r = float(self._particle_size)
        contactOffset, restOffset, particleContactOffset, solidRestOffset, fluidRestOffset = \
            self._compute_particle_system_offsets(r)
        self._fluid_rest_offset = fluidRestOffset

        particle_system = particleUtils.add_physx_particle_system(
            stage,
            self._particleSystemPath,
            contact_offset=contactOffset,
            rest_offset=restOffset,
            particle_contact_offset=particleContactOffset,
            solid_rest_offset=solidRestOffset,
            fluid_rest_offset=fluidRestOffset,
            solver_position_iterations=6,
            simulation_owner=scenePath,
            max_neighborhood=128,
        )

        # smoothing (kept on)
        particleUtils.add_physx_particle_smoothing(stage, self._particleSystemPath, strength=1.0)

        # anisotropy (optional; lower scale to tame artifacts)
        if self._usePointInstancer and self._useAnisotropy:
            particleUtils.add_physx_particle_anisotropy(stage, self._particleSystemPath, scale=self._anisotropy_scale)

        # -------------------- ONLY CHANGE: make it more viscoelastic --------------------
        pbd_particle_material_path = omni.usd.get_stage_next_free_path(stage, "/pbdParticleMaterial", True)
        particleUtils.add_pbd_particle_material(
            stage,
            pbd_particle_material_path,
            cohesion=200.0,          # ↑ more “stick” / filament formation
            viscosity=1000000.0,       # ↑ strong resistance to shear/stretch (syrupy)
            surface_tension=1000.0,   # ↑ tighter clumps / rounded lobes
            friction=500000.0,        # ↑ more drag against surfaces
            damping=0.999,          # ↑ reduces jitter, adds “memory”
        )
        physicsUtils.add_physics_material_to_prim(stage, particle_system.GetPrim(), pbd_particle_material_path)
        # --------------------------------------------------------------------------------

        self._ensure_nozzle_viz(stage)

        self._colors = self.create_colors()
        self._sharedParticlePrim = None
        self._session_sub_layer = None

    # ---------- session layer ----------
    def create_session_layer(self):
        rootLayer = self._stage.GetRootLayer()
        self._session_sub_layer = Sdf.Layer.CreateAnonymous()
        self._stage.GetSessionLayer().subLayerPaths.append(self._session_sub_layer.identifier)
        self._stage.SetEditTarget(Usd.EditTarget(self._session_sub_layer))

    def release_session_layer(self):
        self._stage.GetSessionLayer().subLayerPaths.remove(self._session_sub_layer.identifier)
        self._stage.SetEditTarget(self._stage.GetRootLayer())
        self._session_sub_layer = None

    # ---------- camera backup ----------
    def backup_camera(self):
        cameraPrim = self._stage.GetPrimAtPath("/World/Camera")
        if not cameraPrim:
            return
        self._camera_trans = cameraPrim.GetAttribute("xformOp:translate").Get()
        self._camera_scale = cameraPrim.GetAttribute("xformOp:scale").Get()
        self._camera_rotate = cameraPrim.GetAttribute("xformOp:rotateYXZ").Get()

    def restore_camera(self):
        cameraPrim = self._stage.GetPrimAtPath("/World/Camera")
        if not cameraPrim:
            return
        cameraPrim.GetAttribute("xformOp:translate").Set(self._camera_trans)
        cameraPrim.GetAttribute("xformOp:scale").Set(self._camera_scale)
        cameraPrim.GetAttribute("xformOp:rotateYXZ").Set(self._camera_rotate)

    # ---------- timeline hooks ----------
    def on_timeline_event(self, e):
        if not self._isActive:
            return
        if e.type == int(omni.timeline.TimelineEventType.STOP):
            self._is_running = False
            self._is_paused = False
            self._rng = np.random.default_rng(self._rng_seed)
            self._time = 0.0
            self._emit_accum = 0.0
            self._sharedParticlePrim = None
            self.release_session_layer()
            self.restore_camera()

        if e.type == int(omni.timeline.TimelineEventType.PAUSE):
            self._is_running = False
            self._is_paused = True

        elif e.type == int(omni.timeline.TimelineEventType.PLAY):
            if not self._is_paused:
                self.backup_camera()
                self.create_session_layer()
            self._is_running = True
            self._is_paused = False

    def on_physics_step(self, dt):
        if not self._isActive:
            return
        self._time += dt

        if self._orbit_enabled:
            angle = self._orbit_phase + self._orbit_omega * self._time
            x = self._orbit_center[0] + self._orbit_radius * math.cos(angle)
            y = self._orbit_center[1] + self._orbit_radius * math.sin(angle)
            z = self._orbit_center[2]
            self._nozzle_pos = Gf.Vec3f(x, y, z)

        self._update_nozzle_viz()

    # ---------- emission logic ----------
    def _emit_batch(self, n_particles):
        if n_particles <= 0:
            return
        self.create_shared_particle_prim(self._stage)

        positions = []
        velocities = []

        color_index = int((self._time * 0.5) * self._num_colors) % self._num_colors if self._color_cycle else 0

        for _ in range(n_particles):
            rdisc = self._emit_radius * math.sqrt(self._rng.random())
            theta = 2.0 * math.pi * self._rng.random()
            radial = Gf.Vec3f(rdisc * math.cos(theta), rdisc * math.sin(theta), 0.0)

            jitter = Gf.Vec3f(
                self._rng.uniform(-self._pos_jitter, self._pos_jitter),
                self._rng.uniform(-self._pos_jitter, self._pos_jitter),
                self._rng.uniform(-self._pos_jitter, self._pos_jitter),
            )
            pos = self._nozzle_pos + radial + jitter

            vjit = Gf.Vec3f(
                self._rng.uniform(-self._vel_jitter, self._vel_jitter),
                self._rng.uniform(-self._vel_jitter, self._vel_jitter),
                self._rng.uniform(-self._vel_jitter, self._vel_jitter),
            )
            vel = self._emit_velocity + vjit

            positions.append(pos)
            velocities.append(vel)

        self.add_shared_particles(positions, velocities, color_index)

    def create_shared_particle_prim(self, stage):
        if not self._useSharedParticleSet or self._sharedParticlePrim is not None:
            return

        r = self._particle_size
        rho = 1000.0  # kg/m^3 (water-like)
        mass = (4.0 / 3.0) * math.pi * (r ** 3) * rho

        particlePointsPath = Sdf.Path("/World/Particles")  # under /World

        if self._usePointInstancer:
            self._sharedParticlePrim = particleUtils.add_physx_particleset_pointinstancer(
                stage,
                particlePointsPath,
                [],
                [],
                self._particleSystemPath,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=mass,
                density=0.0,
                num_prototypes=0,
            )
            for i, c in enumerate(self._colors):
                color = Vt.Vec3fArray([c])
                proto_path = str(particlePointsPath) + f"/particlePrototype{i}"
                gprim = UsdGeom.Sphere.Define(stage, Sdf.Path(proto_path))
                gprim.CreateDisplayColorAttr(color)
                gprim.CreateRadiusAttr().Set(r)  # visual radius matches physics radius
                UsdGeom.PointInstancer(self._sharedParticlePrim).GetPrototypesRel().AddTarget(Sdf.Path(proto_path))
                # dummy to keep Hydra happy
                self.add_shared_particles([Gf.Vec3f(-4.0, -1.0 + i * 0.2, -0.5)], [Gf.Vec3f(0.0, 0.0, 0.0)], i)
        else:
            self._sharedParticlePrim = particleUtils.add_physx_particleset_points(
                stage,
                particlePointsPath,
                [],
                [],
                [],
                self._particleSystemPath,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=mass,
                density=0.0,
            )
            self._sharedParticlePrim.CreateDisplayColorAttr().Set(self._colors)
            self._sharedParticlePrim.CreateDisplayColorPrimvar(interpolation="vertex")
            self._sharedParticlePrim.GetDisplayColorPrimvar().CreateIndicesAttr().Set([])

        self._sharedParticlePrim.GetPrim().CreateAttribute(
            "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
        ).Set(self._max_particles)

    def update(self, stage, dt, viewport, physxIFace):
        if not (self._isActive and self._is_running and self._emit_enabled):
            return

        self.backup_camera()

        if self._sharedParticlePrim is not None:
            pi = UsdGeom.PointInstancer(self._sharedParticlePrim)
            if pi:
                curr = pi.GetPositionsAttr().Get()
                curr_count = 0 if curr is None else len(curr)
                if curr_count >= self._max_particles:
                    return

        self._emit_accum += dt * self._emit_rate_particles_per_second
        emit_now = int(self._emit_accum)
        self._emit_accum -= emit_now

        if self._sharedParticlePrim is not None and emit_now > 0:
            pi = UsdGeom.PointInstancer(self._sharedParticlePrim)
            if pi:
                curr = pi.GetPositionsAttr().Get()
                curr_count = 0 if curr is None else len(curr)
                remaining = max(self._max_particles - curr_count, 0)
                emit_now = min(emit_now, remaining)

        if emit_now > 0:
            self._emit_batch(emit_now)

    def on_shutdown(self):
        self._isActive = False
        self._stage = None
        super().on_shutdown()