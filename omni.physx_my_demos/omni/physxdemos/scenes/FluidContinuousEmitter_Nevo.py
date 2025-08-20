import math
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, Vt, UsdPhysics, PhysxSchema
from omni.physx.scripts import utils
from omni.physx.scripts import physicsUtils, particleUtils
import omni.physxdemos as demo
import omni.physx.bindings._physx as physx_settings_bindings
import omni.timeline
import numpy as np


class FluidBallEmitterDemo(demo.AsyncDemoBase):
    title = "Continuous Fluid Emitter"
    category = demo.Categories.PARTICLES
    short_description = "PBD fluid continuously emitted from a nozzle"
    description = "Continuously emits PBD fluid particles with optional smoothing/anisotropy."

    # Keep UI params compatible with the original demo
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

        # continuous emission controls (tweak freely)
        self._emit_enabled = True
        self._emit_rate_particles_per_second = 1500            # particles / s
        self._emit_radius = 0.35                                # nozzle radius (m)
        self._emit_velocity = Gf.Vec3f(10.0, 10.0, 0.0)         # initial velocity (m/s)
        self._vel_jitter = 0.5                                  # +/- m/s jitter per axis
        self._pos_jitter = 0.02                                 # +/- m position jitter per axis
        self._emit_accum = 0.0                                  # fractional particle accumulator

        # capacity (must be large enough for your runtime)
        self._max_particles = 100_000

        # visuals / colors
        self._num_colors = 20
        self._color_cycle = True

        # USD / stage handles
        self._isActive = True
        self._stage = None
        self._sharedParticlePrim = None
        self._session_sub_layer = None

        # fluid spacing (set in create)
        self._fluid_rest_offset = 0.05

        # nozzle in world space (above the demo room floor)
        self._nozzle_pos = Gf.Vec3f(-6.0, -6.0, 1.2)

    # ---------- utility: colors ----------
    def create_colors(self):
        fractions = np.linspace(0.0, 1.0, self._num_colors)
        return [self.create_color(frac) for frac in fractions]

    def create_color(self, frac):
        # simple HSL->RGB-ish rainbow, like the original
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
        rgb = luminosity * linter
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

        # write sim positions first if smoothing enabled (avoids a frame of mismatch)
        if self._useSmoothing:
            simPointsAttr = particleSet.GetSimulationPointsAttr()
            if not simPointsAttr.HasAuthoredValue():
                simPointsAttr.Set(Vt.Vec3fArray([]))
            self.extend_array_attribute(simPointsAttr, positions_list)

        if pointInstancer:
            self.extend_array_attribute(pointInstancer.GetPositionsAttr(), positions_list)
            self.extend_array_attribute(pointInstancer.GetVelocitiesAttr(), velocities_list)
            self.extend_array_attribute(pointInstancer.GetProtoIndicesAttr(), [color_index] * len(positions_list))
            self.extend_array_attribute(pointInstancer.GetOrientationsAttr(), [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * len(positions_list))
            self.extend_array_attribute(pointInstancer.GetScalesAttr(), [Gf.Vec3f(1.0)] * len(positions_list))
        elif points:
            self.extend_array_attribute(points.GetPointsAttr(), positions_list)
            self.extend_array_attribute(points.GetVelocitiesAttr(), velocities_list)
            self.extend_array_attribute(points.GetWidthsAttr(), [2 * self._fluid_rest_offset] * len(positions_list))
            primVars = points.GetDisplayColorPrimvar()
            primVarsIndicesAttr = primVars.GetIndicesAttr()
            self.extend_array_attribute(primVarsIndicesAttr, [color_index] * len(positions_list))

    def create_shared_particle_prim(self, stage):
        if not self._useSharedParticleSet or self._sharedParticlePrim is not None:
            return

        particlePointsPath = Sdf.Path("/particles")

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
                particle_mass=0.001,
                density=0.0,
                num_prototypes=0,
            )

            # Create prototypes for color indices
            for i, c in enumerate(self._colors):
                color = Vt.Vec3fArray([c])
                proto_path = str(particlePointsPath) + f"/particlePrototype{i}"
                gprim = UsdGeom.Sphere.Define(stage, Sdf.Path(proto_path))
                gprim.CreateDisplayColorAttr(color)
                gprim.CreateRadiusAttr().Set(self._fluid_rest_offset)
                UsdGeom.PointInstancer(self._sharedParticlePrim).GetPrototypesRel().AddTarget(Sdf.Path(proto_path))
                # add dummy particle per prototype to keep Hydra happy
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
                particle_mass=0.001,
                density=0.0,
            )
            self._sharedParticlePrim.CreateDisplayColorAttr().Set(self._colors)
            self._sharedParticlePrim.CreateDisplayColorPrimvar(interpolation="vertex")
            self._sharedParticlePrim.GetDisplayColorPrimvar().CreateIndicesAttr().Set([])

        # IMPORTANT: set particle capacity for continuous emission
        self._sharedParticlePrim.GetPrim().CreateAttribute(
            "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
        ).Set(self._max_particles)

    # ---------- scene setup ----------
    def create(self, stage, Single_Particle_Set, Use_Instancer):
        self._stage = stage
        self._setup_callbacks()

        # force shared particle set for continuous emission
        self._useSharedParticleSet = True if Single_Particle_Set else True
        self._usePointInstancer = True if Use_Instancer else True
        self._useSmoothing = True
        self._useAnisotropy = True

        defaultPrimPath, scene = demo.setup_physics_scene(self, stage, metersPerUnit=1.0)
        scenePath = defaultPrimPath + "/physicsScene"

        # Particle System
        self._particleSystemPath = Sdf.Path("/particleSystem0")

        # conservative spacing/offsets
        particleSpacing = 0.18
        restOffset = particleSpacing * 0.9
        solidRestOffset = restOffset
        fluidRestOffset = restOffset * 0.6
        particleContactOffset = max(solidRestOffset + 0.005, fluidRestOffset / 0.6)
        contactOffset = restOffset + 0.005
        self._fluid_rest_offset = fluidRestOffset

        particle_system = particleUtils.add_physx_particle_system(
            stage,
            self._particleSystemPath,
            contact_offset=contactOffset,
            rest_offset=restOffset,
            particle_contact_offset=particleContactOffset,
            solid_rest_offset=solidRestOffset,
            fluid_rest_offset=fluidRestOffset,
            solver_position_iterations=4,
            simulation_owner=scenePath,
            max_neighborhood=96,
        )

        if self._useSmoothing:
            particleUtils.add_physx_particle_smoothing(stage, self._particleSystemPath, strength=1.0)

        if self._usePointInstancer and self._useAnisotropy:
            particleUtils.add_physx_particle_anisotropy(stage, self._particleSystemPath, scale=1.0)

        # pbd material
        pbd_particle_material_path = omni.usd.get_stage_next_free_path(stage, "/pbdParticleMaterial", True)
        particleUtils.add_pbd_particle_material(
            stage,
            pbd_particle_material_path,
            cohesion=5,
            viscosity=1000,
            surface_tension=0.02,
            friction=1000,
            damping=0.99,
        )
        physicsUtils.add_physics_material_to_prim(stage, particle_system.GetPrim(), pbd_particle_material_path)

        # room & colors
        demo.get_demo_room(self, stage)
        self._colors = self.create_colors()

        self._sharedParticlePrim = None
        self._session_sub_layer = None

    # ---------- session layer for runtime edits ----------
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

    # ---------- emission logic ----------
    def _emit_batch(self, n_particles):
        if n_particles <= 0:
            return
        # ensure the shared particle set exists
        self.create_shared_particle_prim(self._stage)

        positions = []
        velocities = []

        # color selection
        if self._color_cycle:
            color_index = int((self._time * 0.5) * self._num_colors) % self._num_colors
        else:
            color_index = 0

        for _ in range(n_particles):
            # sample point in a disc (XY), jitter in Z
            r = self._emit_radius * math.sqrt(self._rng.random())
            theta = 2.0 * math.pi * self._rng.random()
            radial = Gf.Vec3f(r * math.cos(theta), r * math.sin(theta), 0.0)

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

    def update(self, stage, dt, viewport, physxIFace):
        if not (self._isActive and self._is_running and self._emit_enabled):
            return

        self.backup_camera()

        # capacity guard
        if self._sharedParticlePrim is not None:
            # read current count from instancer if possible (cheap heuristic: use positions array length)
            pi = UsdGeom.PointInstancer(self._sharedParticlePrim)
            if pi:
                pos_attr = pi.GetPositionsAttr()
                curr = pos_attr.Get()
                curr_count = 0 if curr is None else len(curr)
                if curr_count >= self._max_particles:
                    return

        # accumulate particles to emit
        self._emit_accum += dt * self._emit_rate_particles_per_second
        emit_now = int(self._emit_accum)
        self._emit_accum -= emit_now

        # clamp to remaining capacity if known
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