#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s03_s04_complete.py  --  Continue the S03 house model for the S03-S04 activity
================================================================================

Target stack : Blender 4.5 LTS + Bonsai 0.8.x + IfcOpenShell >= 0.8.0
               (it also runs in plain Python with `pip install ifcopenshell`)

WHAT IT DOES (all on a COPY - the source file is never modified)
  1. Copies  s03-discussion.ifc  ->  s03-s04.ifc  and edits the copy.
  2. Sets storey elevations (GF 0.0 / 2F 3.0 / Roof 6.0) WITHOUT moving any
     existing element in world space.
  3. 2F walls + partitions  (existing "Wall 2F" type), butt-jointed.
  4. 2F doors and windows   (existing 2F door/window TYPES), each hosted in a
     voided wall.
  5. Stair GF -> 2F : two IfcStairFlight + landing (16 risers over 3.0 m).
  6. Stair opening cut through the existing 2F slab.
  7. Stair railings (GF) + 2F barriers (stair opening, balcony).
  8. Roof : flat "Slab Roof" at wall-top level + hip IfcRoof (existing "Roof" type).
  9. Validates, writes s03-s04.ifc, and (only if running inside Blender with
     Bonsai) loads it and saves s03-s04.blend.

HOW TO RUN  -- see the README block at the bottom of this file, or:
  Inside Blender : Scripting workspace -> Open this file -> edit SOURCE_IFC
                   below if needed -> Run Script.
  Command line   : blender --background --python s03_s04_complete.py -- \
                       --source "D:/s03-discussion.ifc"
                   python s03_s04_complete.py --source D:/s03-discussion.ifc

Every dimension that was READ FROM A DRAWING (not given as a dimension line)
is tagged  [MEASURED]  in the constants below, and every judgement call is
tagged  [ASSUMED].  Nothing here has been checked against a site survey.
"""

import os
import sys
import json
import math
import shutil
import traceback

import numpy as np

# --------------------------------------------------------------------------- #
#  CONFIGURATION
# --------------------------------------------------------------------------- #

# Default location of the starting project.  Override with --source or the
# S04_SOURCE environment variable.  The first existing candidate wins.
SOURCE_CANDIDATES = [
    r"D:\s03-discussion.ifc",
    "s03-discussion.ifc",
]
OUTPUT_IFC_NAME = "s03-s04.ifc"
OUTPUT_BLEND_NAME = "s03-s04.blend"

# --- storeys -------------------------------------------------------------- #
ELEV_GF, ELEV_2F, ELEV_ROOF = 0.0, 3.0, 6.0     # matches the activity slides
STOREY_HEIGHT = ELEV_2F - ELEV_GF                # 3.0 m  (existing 2F slab top)

# --- walls ---------------------------------------------------------------- #
WALL_T = 0.10                                    # same as every existing wall / "Wall 2F" layer
WALL_H = ELEV_ROOF - ELEV_2F                     # 3.0 m, 2F floor -> roof storey

# --- stair  (plan: 15 numbered treads, two parallel flights, landing "8") -- #
RISERS_TOTAL = 16                                # 15 numbered treads + final riser onto 2F
RISERS_PER_FLIGHT = RISERS_TOTAL // 2            # 8  (7 treads + 1 riser onto landing)
TREADS_PER_FLIGHT = RISERS_PER_FLIGHT - 1        # 7
RISER_H = STOREY_HEIGHT / RISERS_TOTAL           # 0.1875 m
TREAD_D = 0.235                                  # [MEASURED] ~0.236 m from plan
STAIR_X0 = 1.675                                 # [MEASURED] west end of flights (+-0.03)
STAIR_Y0 = 0.47                                  # [MEASURED] 6025 mm chain from north face
STAIR_Y1 = 2.29                                  # [MEASURED] 4210 mm chain from north face
FLIGHT_W = 0.80                                  # [ASSUMED]  clear width of each flight
EAST_WALL_INNER_X = 4.127                        # existing east wall inner face
LANDING_T = 0.20                                 # [ASSUMED]
# NOTE: the plan only draws tread "8" ~0.2 m deep, which is not a usable
# landing.  The landing here runs from the end of the flights to the east wall.

# --- railings / roof ------------------------------------------------------ #
RAIL_H = 0.90                                    # [ASSUMED] guard height
RAIL_DIA = 0.04
RAIL_POST_SPACING = 0.90
ROOF_OVERHANG = 0.30                             # [ASSUMED] eave overhang beyond 2F slab
ROOF_ANGLE = 0.1745329251994329                  # 10 deg - value stored in the file's roof TYPE
ROOF_THICK = 0.10                                # value stored in the file's roof TYPE

# --- wall layout ----------------------------------------------------------- #
# Each wall is a rectangle in WORLD coordinates (metres) + running direction.
#   run='x': wall runs along +X, reference line on the SOUTH face (ymin)
#   run='y': wall runs along +Y, reference line on the EAST  face (xmax)
# This is exactly how the existing GF walls are built (thickness grows toward
# the wall's local +Y).  Rectangles are chosen so walls BUTT against each other
# (no overlapping volumes).
#
# Anchors that come from the file itself:
#   * 2F slab outline  : x -3.273..4.227, y -3.100..6.500, NW corner notched
#                        (x < 0.227 and y > 5.550 has NO slab)
#   * B-grid wall      : x 0.227..0.327  (same as existing GF walls)
#   * grid-3 partition : y 2.538..2.638  (same as existing GF wall below)
# Everything else is [MEASURED] from second_floor.svg (+-0.03 m) and cross
# checked against the dimension chains printed on the sheet.
WALLS = [
    # tag      run   xmin    xmax    ymin    ymax
    ("E",      "y",  4.127,  4.227, -3.100,  6.500),   # east exterior
    ("N",      "x",  0.227,  4.127,  6.400,  6.500),   # north exterior (T&B 2)
    ("B",      "y",  0.227,  0.327, -3.100,  6.400),   # grid-B wall
    ("BR1-N",  "x", -3.273,  0.227,  5.450,  5.550),   # Bedroom 1 north (975 chain)
    ("W",      "y", -3.273, -3.173, -1.530,  5.450),   # west exterior
    ("MBR-S",  "x", -3.273,  0.227, -1.630, -1.530),   # Master south (8120 chain)
    ("G3",     "x", -3.173,  0.227,  2.538,  2.638),   # BR1 / Master partition
    ("TB2-S",  "x",  0.327,  4.127,  4.890,  4.990),   # T&B 2 south partition
    ("BR2-N",  "x",  0.327,  4.127,  0.360,  0.460),   # Bedroom 2 north (6025 chain)
    ("NOTCH",  "x",  0.327,  1.950, -2.195, -2.095),   # Bedroom 2 / balcony partition
    ("NOTCH2", "y",  1.950,  2.050, -3.000, -2.095),   # Bedroom 2 return wall
    ("S",      "x",  1.950,  4.127, -3.100, -3.000),   # Bedroom 2 south exterior
]

# Openings.  `type` = name of an EXISTING IfcDoorType / IfcWindowType in the
# discussion file (their widths already match the plan).  `at` = centre of the
# opening in WORLD coordinates along the host wall (x for 'x' walls, y for 'y').
DOORS = [
    # wall      type                  at      note
    ("B",      "Bedroom1 2F",          3.220), # BR1 door, hinge/arc read from plan
    ("B",      "Master 2F",            2.000), # Master door
    ("TB2-S",  "Bathroom 2F",          0.975), # T&B 2 door
    ("BR2-N",  "Bedroom2 2F",          0.975), # Bedroom 2 door #1
    ("NOTCH",  "Balcony 2F",           0.975), # Bedroom 2 -> balcony door
    ("MBR-S",  "Master-Balcony-2F",   -0.900), # Master -> balcony sliding door (2.0 m)
]
WINDOWS = [
    ("BR1-N",  "BR1-2F-2",            -0.830), # 1.209 m, matches plan
    ("W",      "BR1-2F-1",             3.600),
    ("W",      "Master-2F-1",          1.510),
    ("W",      "Master-2F-2",         -1.050),
    ("S",      "BR2-2F",               2.775),
]
DOOR_SILL = 0.0
WINDOW_SILL = 1.0                                # same sill as existing GF windows
DOOR_OPENING_Y = 0.015                           # same offsets the existing GF openings use
WINDOW_OPENING_Y = 0.0

EXPECTED_2F = dict(IfcWall=len(WALLS), IfcDoor=len(DOORS), IfcWindow=len(WINDOWS))

# --------------------------------------------------------------------------- #
#  LOGGING
# --------------------------------------------------------------------------- #

_WARNINGS, _ERRORS = [], []


def info(msg):
    print(f"[INFO] {msg}")


def ok(msg):
    print(f"[ OK ] {msg}")


def warn(msg):
    _WARNINGS.append(msg)
    print(f"[WARN] {msg}")


def err(msg):
    _ERRORS.append(msg)
    print(f"[FAIL] {msg}")


# --------------------------------------------------------------------------- #
#  ENVIRONMENT
# --------------------------------------------------------------------------- #

def in_blender():
    try:
        import bpy  # noqa: F401
        return True
    except ImportError:
        return False


def check_environment():
    """Report Blender / Bonsai / IfcOpenShell versions and enforce a minimum."""
    info("---- environment ----")
    info(f"Python        : {sys.version.split()[0]}")
    if in_blender():
        import bpy
        info(f"Blender       : {bpy.app.version_string}")
        bonsai_ver = None
        for mod in ("bonsai", "blenderbim"):
            try:
                m = __import__(mod)
                bonsai_ver = f"{mod} {'.'.join(map(str, m.bl_info['version']))}"
                break
            except Exception:
                continue
        if bonsai_ver:
            info(f"Bonsai        : {bonsai_ver}")
        else:
            warn("Bonsai add-on not found - IFC will be written but .blend cannot be created.")
    else:
        info("Blender       : not running inside Blender (headless IfcOpenShell mode)")
    try:
        import ifcopenshell
    except ImportError:
        err("ifcopenshell is not importable. Inside Blender enable the Bonsai add-on; "
            "outside Blender run: pip install ifcopenshell")
        raise SystemExit(1)
    ver = ifcopenshell.version
    info(f"IfcOpenShell  : {ver}")
    try:
        major, minor = [int(x) for x in ver.replace("v", "").split(".")[:2]]
        if (major, minor) < (0, 8):
            warn("IfcOpenShell < 0.8 detected: API names (feature.*) may differ; script has fallbacks "
                 "but is untested on this version.")
    except Exception:
        pass
    return ver


def resolve_args():
    """--source PATH  --out-dir DIR  --no-blender   (args after '--' inside Blender)."""
    argv = sys.argv[1:]
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    opts = {"source": os.environ.get("S04_SOURCE"), "out_dir": None,
            "blender": os.environ.get("S04_SKIP_BLENDER", "") == ""}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--source" and i + 1 < len(argv):
            opts["source"] = argv[i + 1]; i += 2
        elif a == "--out-dir" and i + 1 < len(argv):
            opts["out_dir"] = argv[i + 1]; i += 2
        elif a == "--no-blender":
            opts["blender"] = False; i += 1
        else:
            i += 1
    if not opts["source"]:
        here = os.path.dirname(os.path.abspath(globals().get("__file__", os.getcwd())))
        cands = SOURCE_CANDIDATES + [os.path.join(here, "s03-discussion.ifc")]
        for c in cands:
            if os.path.isfile(c):
                opts["source"] = c
                break
    if not opts["source"] or not os.path.isfile(opts["source"]):
        err("Source IFC not found. Pass --source <path to s03-discussion.ifc> "
            "or edit SOURCE_CANDIDATES at the top of the script.")
        raise SystemExit(1)
    opts["source"] = os.path.abspath(opts["source"])
    opts["out_dir"] = os.path.abspath(opts["out_dir"] or os.path.dirname(opts["source"]))
    return opts


# --------------------------------------------------------------------------- #
#  SMALL HELPERS
# --------------------------------------------------------------------------- #

def mat4(origin, xdir=(1, 0, 0), zdir=(0, 0, 1)):
    """Right-handed 4x4 placement: local X=xdir, local Z=zdir, Y=Z x X."""
    x = np.array(xdir, float); x /= np.linalg.norm(x)
    z = np.array(zdir, float); z /= np.linalg.norm(z)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, origin
    return m


def world(el):
    import ifcopenshell.util.placement as upl
    return upl.get_local_placement(el.ObjectPlacement) if el.ObjectPlacement else None


def contained(storey):
    out = []
    for rel in getattr(storey, "ContainsElements", []) or []:
        out.extend(rel.RelatedElements)
    return out


def by_name(f, ifc_class, name):
    hits = [e for e in f.by_type(ifc_class) if e.Name == name]
    if not hits:
        raise LookupError(f"{ifc_class} named '{name}' not found in the source file")
    return hits[0]


def signed_volume(verts, faces):
    v = np.array(verts, float)
    vol = 0.0
    for fc in faces:
        for i in range(1, len(fc) - 1):
            a, b, c = v[fc[0]], v[fc[i]], v[fc[i + 1]]
            vol += np.dot(a, np.cross(b, c)) / 6.0
    return vol


def outward(verts, faces):
    """Flip all faces if the mesh has negative volume so normals point outward."""
    if signed_volume(verts, faces) < 0:
        faces = [list(reversed(fc)) for fc in faces]
    return faces


# --------------------------------------------------------------------------- #
#  MODEL BUILDER
# --------------------------------------------------------------------------- #

class Builder:
    def __init__(self, ifc_path):
        import ifcopenshell
        import ifcopenshell.api.root, ifcopenshell.api.geometry, ifcopenshell.api.spatial   # noqa
        import ifcopenshell.api.type, ifcopenshell.api.pset, ifcopenshell.api.attribute      # noqa
        import ifcopenshell.api.material, ifcopenshell.api.feature                            # noqa
        self.ifcopenshell = ifcopenshell
        self.api = ifcopenshell.api
        self.f = ifcopenshell.open(ifc_path)
        if self.f.schema != "IFC4":
            warn(f"File schema is {self.f.schema}; script was written for IFC4.")
        self.path = ifc_path
        self.storeys = {s.Name: s for s in self.f.by_type("IfcBuildingStorey")}
        for need in ("GF", "2F", "Roof"):
            if need not in self.storeys:
                raise LookupError(f"Storey '{need}' missing in source file")
        ctxs = self.f.by_type("IfcGeometricRepresentationSubContext")
        self.body = next(c for c in ctxs if c.ContextIdentifier == "Body" and c.TargetView == "MODEL_VIEW")
        self.axis = next((c for c in ctxs if c.ContextIdentifier == "Axis"), None)
        self.walls = {}          # tag -> IfcWall
        self.wall_data = {}       # tag -> plain Python wall metadata (IFC entities reject custom attrs)
        self.created = []        # every new product (for the report)

    # ---- generic ---------------------------------------------------------- #
    def new(self, ifc_class, name, storey, matrix, predefined=None):
        e = self.api.root.create_entity(self.f, ifc_class=ifc_class, name=name, predefined_type=predefined) \
            if predefined else self.api.root.create_entity(self.f, ifc_class=ifc_class, name=name)
        self.api.spatial.assign_container(self.f, products=[e], relating_structure=storey)
        self.api.geometry.edit_object_placement(self.f, e, matrix)
        self.created.append(e)
        return e

    def set_rep(self, product, rep):
        self.api.geometry.assign_representation(self.f, product=product, representation=rep)

    def add_feature(self, opening, host):
        fn = getattr(self.api.feature, "add_feature", None)
        if fn is None:                                      # IfcOpenShell < 0.8 fallback
            import ifcopenshell.api.void
            return self.api.void.add_opening(self.f, opening=opening, element=host)
        return fn(self.f, feature=opening, element=host)

    def add_filling(self, opening, element):
        fn = getattr(self.api.feature, "add_filling", None)
        if fn is None:
            import ifcopenshell.api.void
            return self.api.void.add_filling(self.f, opening=opening, element=element)
        return fn(self.f, opening=opening, element=element)

    def local_placement(self, parent_placement, loc=(0, 0, 0), z=None, x=None):
        f = self.f
        pt = f.createIfcCartesianPoint([float(v) for v in loc])
        ax = f.createIfcDirection([float(v) for v in z]) if z else None
        rx = f.createIfcDirection([float(v) for v in x]) if x else None
        return f.createIfcLocalPlacement(parent_placement, f.createIfcAxis2Placement3D(pt, ax, rx))

    def opening_rep_wall(self, w, h):
        """Opening body identical in construction to the file's existing openings:
        w x h profile, 1.2 m deep, centred on the wall reference plane."""
        f = self.f
        pts = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h), (0.0, 0.0)]
        poly = f.createIfcPolyline([f.createIfcCartesianPoint(p) for p in pts])
        prof = f.createIfcArbitraryClosedProfileDef("AREA", None, poly)
        pos = f.createIfcAxis2Placement3D(f.createIfcCartesianPoint((0.0, -0.6, 0.0)),
                                          f.createIfcDirection((0.0, -1.0, 0.0)),
                                          f.createIfcDirection((1.0, 0.0, 0.0)))
        solid = f.createIfcExtrudedAreaSolid(prof, pos, f.createIfcDirection((0.0, 0.0, -1.0)), 1.2)
        return f.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [solid])

    def opening_rep_box_z(self, dx, dy, z0, depth):
        """Vertical box opening (for slabs): dx x dy footprint, from z0, `depth` tall."""
        f = self.f
        pts = [(0.0, 0.0), (dx, 0.0), (dx, dy), (0.0, dy), (0.0, 0.0)]
        poly = f.createIfcPolyline([f.createIfcCartesianPoint(p) for p in pts])
        prof = f.createIfcArbitraryClosedProfileDef("AREA", None, poly)
        pos = f.createIfcAxis2Placement3D(f.createIfcCartesianPoint((0.0, 0.0, z0)), None, None)
        solid = f.createIfcExtrudedAreaSolid(prof, pos, f.createIfcDirection((0.0, 0.0, 1.0)), depth)
        return f.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [solid])

    # ---- step 1: storeys --------------------------------------------------- #
    def fix_storeys(self):
        info("Step 1: storey elevations / placements")
        pre = {e.GlobalId: world(e) for e in self.f.by_type("IfcElement") if e.ObjectPlacement}
        for name, z in (("GF", ELEV_GF), ("2F", ELEV_2F), ("Roof", ELEV_ROOF)):
            s = self.storeys[name]
            self.api.attribute.edit_attributes(self.f, product=s, attributes={"Elevation": float(z)})
            # edit_object_placement(should_transform_children=False) rewrites children's
            # local placements so their WORLD position does not change.
            self.api.geometry.edit_object_placement(self.f, s, mat4((0, 0, z)),
                                                    should_transform_children=False)
            ok(f"storey {name}: Elevation={z}")
        moved = [gid for gid, m in pre.items()
                 if not np.allclose(m, world(self.f.by_guid(gid)), atol=1e-6)]
        if moved:
            err(f"{len(moved)} existing element(s) moved in world space while re-elevating storeys!")
        else:
            ok(f"{len(pre)} existing elements kept their world position")
        return pre

    # ---- step 2/3: walls, doors, windows ---------------------------------- #
    def build_walls(self):
        if any(e.is_a("IfcWall") for e in contained(self.storeys["2F"])):
            warn("2F already contains walls - skipping wall creation to avoid duplicates.")
            return
        info("Step 2: 2F walls")
        wt = by_name(self.f, "IfcWallType", "Wall 2F")
        for tag, run, x0, x1, y0, y1 in WALLS:
            if run == "x":
                origin, d, L = (x0, y0, ELEV_2F), (1, 0, 0), x1 - x0
            else:
                origin, d, L = (x1, y0, ELEV_2F), (0, 1, 0), y1 - y0
            wall = self.new("IfcWall", f"2F Wall {tag}", self.storeys["2F"], mat4(origin, d))
            self.api.type.assign_type(self.f, related_objects=[wall], relating_type=wt)
            self.set_rep(wall, self.api.geometry.add_wall_representation(
                self.f, context=self.body, length=L, height=WALL_H, thickness=WALL_T,
                offset=0.0, direction_sense="POSITIVE"))
            if self.axis is not None:
                self.set_rep(wall, self.api.geometry.add_axis_representation(
                    self.f, context=self.axis, axis=[(0.0, 0.0), (L, 0.0)]))
            ps = self.api.pset.add_pset(self.f, product=wall, name="EPset_Parametric")
            self.api.pset.edit_pset(self.f, pset=ps,
                                    properties={"LayerSetDirection": "AXIS2", "Engine": "Bonsai.DumbLayer2"})
            self.wall_data[tag] = dict(run=run, x0=x0, x1=x1, y0=y0, y1=y1, L=L)
            self.walls[tag] = wall
        ok(f"created {len(self.walls)} walls")

    def _type_dims(self, t):
        import ifcopenshell.util.element as uel
        ps = uel.get_psets(t)
        key = "BBIM_Door" if t.is_a("IfcDoorType") else "BBIM_Window"
        data = ps.get(key, {}).get("Data")
        d = json.loads(data) if isinstance(data, str) else (data or {})
        return float(d["overall_width"]), float(d["overall_height"])

    def _host_fill(self, tag, type_class, type_name, at, sill, oy, label):
        wall = self.walls[tag]
        s = self.wall_data[tag]
        t = by_name(self.f, type_class, type_name)
        w, h = self._type_dims(t)
        s_c = (at - s["x0"]) if s["run"] == "x" else (at - s["y0"])
        left = s_c - w / 2.0
        if left < 0.0 or left + w > s["L"]:
            err(f"{label}: opening does not fit inside wall '{tag}' (s={left:.3f}..{left + w:.3f}, L={s['L']:.3f})")
            return None
        cls = "IfcDoor" if type_class == "IfcDoorType" else "IfcWindow"
        # opening -> relative to the wall (same as the file's existing openings)
        op = self.api.root.create_entity(self.f, ifc_class="IfcOpeningElement", predefined_type="OPENING", name="Opening")
        op.ObjectPlacement = self.local_placement(wall.ObjectPlacement, (left, oy, sill))
        self.set_rep(op, self.opening_rep_wall(w, h))
        self.add_feature(op, wall)
        # door/window -> relative to the opening
        el = self.api.root.create_entity(self.f, ifc_class=cls, name=label)
        self.api.spatial.assign_container(self.f, products=[el], relating_structure=self.storeys["2F"])
        el.ObjectPlacement = self.local_placement(op.ObjectPlacement, (0.0, 0.0, 0.0))
        self.api.type.assign_type(self.f, related_objects=[el], relating_type=t)   # maps the type's geometry
        attrs = {"OverallWidth": w, "OverallHeight": h}
        if cls == "IfcDoor":
            attrs["OperationType"] = t.OperationType
        else:
            attrs["PartitioningType"] = t.PartitioningType
        self.api.attribute.edit_attributes(self.f, product=el, attributes=attrs)
        self.add_filling(op, el)
        self.created.extend([el, op])
        return el

    def build_doors_windows(self):
        if any(e.is_a("IfcDoor") or e.is_a("IfcWindow") for e in contained(self.storeys["2F"])):
            warn("2F already contains doors/windows - skipping to avoid duplicates.")
            return
        info("Step 3: 2F doors and windows (existing types)")
        n = 0
        for tag, tname, at in DOORS:
            n += bool(self._host_fill(tag, "IfcDoorType", tname, at, DOOR_SILL, DOOR_OPENING_Y, f"2F Door {tname}"))
        m = 0
        for tag, tname, at in WINDOWS:
            m += bool(self._host_fill(tag, "IfcWindowType", tname, at, WINDOW_SILL, WINDOW_OPENING_Y,
                                      f"2F Window {tname}"))
        ok(f"created {n} doors and {m} windows")

    # ---- step 4: stair -------------------------------------------------------- #
    def _flight_profile(self):
        T, rh, n = TREAD_D, RISER_H, TREADS_PER_FLIGHT
        pts = [(0.0, 0.0)]
        for k in range(1, n + 1):
            pts.append(((k - 1) * T, k * rh))
            pts.append((k * T, k * rh))
        pts.append((n * T, (n + 1) * rh))      # last riser up to the landing / 2F floor
        pts.append((n * T, n * rh))            # square end of the soffit
        return pts                             # closes back to (0,0): straight sloped soffit

    def build_stair(self):
        gf = self.storeys["GF"]
        if any(e.is_a("IfcStairFlight") for e in self.f.by_type("IfcStairFlight")):
            warn("A stair flight already exists - skipping stair creation.")
            return None
        info("Step 4: stair (2 flights + landing)")
        run = TREADS_PER_FLIGHT * TREAD_D
        x_end = STAIR_X0 + run                                     # where flights meet the landing
        y_f1 = (STAIR_Y0, STAIR_Y0 + FLIGHT_W)                     # south flight
        y_f2 = (STAIR_Y1 - FLIGHT_W, STAIR_Y1)                     # north flight
        gap = y_f2[0] - y_f1[1]
        info(f"  flight run {run:.3f} m, riser {RISER_H * 1000:.1f} mm, tread {TREAD_D * 1000:.0f} mm, "
             f"well gap {gap:.3f} m, landing x {x_end:.3f}..{EAST_WALL_INNER_X:.3f}")
        if gap < 0:
            err("Stair flights overlap - check STAIR_Y0/Y1 and FLIGHT_W")
        prof = self._flight_profile()

        def make_flight(name, origin, xdir, zdir, z_start):
            fl = self.new("IfcStairFlight", name, gf, mat4(origin, xdir, zdir), predefined="STRAIGHT")
            self.set_rep(fl, self.api.geometry.add_slab_representation(
                self.f, context=self.body, depth=FLIGHT_W, direction_sense="POSITIVE",
                offset=0.0, polyline=prof))
            self.api.attribute.edit_attributes(self.f, product=fl, attributes={
                "NumberOfRisers": RISERS_PER_FLIGHT, "NumberOfTreads": TREADS_PER_FLIGHT,
                "RiserHeight": RISER_H, "TreadLength": TREAD_D})
            return fl

        # Flight 1: starts at the west end (z=0), climbs EAST. Profile plane = world XZ,
        # extruded toward -Y, so the origin sits on the flight's north edge.
        f1 = make_flight("Stair Flight 1", (STAIR_X0, y_f1[1], ELEV_GF), (1, 0, 0), (0, -1, 0), ELEV_GF)
        # Flight 2: starts at the landing (z = half height), climbs WEST, extruded toward +Y.
        z_mid = ELEV_GF + RISERS_PER_FLIGHT * RISER_H
        f2 = make_flight("Stair Flight 2", (x_end, y_f2[0], z_mid), (-1, 0, 0), (0, 1, 0), z_mid)

        # Landing (IfcSlab / LANDING)
        L = self.new("IfcSlab", "Stair Landing", gf, mat4((x_end, STAIR_Y0, z_mid - LANDING_T)),
                     predefined="LANDING")
        self.set_rep(L, self.api.geometry.add_slab_representation(
            self.f, context=self.body, depth=LANDING_T, direction_sense="POSITIVE", offset=0.0,
            polyline=[(0.0, 0.0), (EAST_WALL_INNER_X - x_end, 0.0),
                      (EAST_WALL_INNER_X - x_end, STAIR_Y1 - STAIR_Y0), (0.0, STAIR_Y1 - STAIR_Y0)]))
        ok("created 2 stair flights + landing")
        return dict(x_end=x_end, y_f1=y_f1, y_f2=y_f2, z_mid=z_mid)

    def build_stair_opening(self):
        slab = next((e for e in contained(self.storeys["2F"]) if e.is_a("IfcSlab")), None)
        if slab is None:
            err("No 2F slab found - cannot cut the stair opening")
            return
        if slab.HasOpenings:
            warn("2F slab already has an opening - skipping stair opening.")
            return
        info("Step 5: stair opening in the 2F slab")
        x0, x1, y0, y1 = STAIR_X0, EAST_WALL_INNER_X, STAIR_Y0, STAIR_Y1
        op = self.api.root.create_entity(self.f, ifc_class="IfcOpeningElement", predefined_type="OPENING",
                                         name="Stair Opening")
        # place relative to the slab: local = inverse(slab world) * world corner
        inv = np.linalg.inv(world(slab))
        loc = inv @ np.array([x0, y0, ELEV_2F, 1.0])
        op.ObjectPlacement = self.local_placement(slab.ObjectPlacement, tuple(loc[:3]))
        self.set_rep(op, self.opening_rep_box_z(x1 - x0, y1 - y0, -0.25, 0.5))
        self.add_feature(op, slab)
        self.created.append(op)
        ok(f"cut {x1 - x0:.3f} x {y1 - y0:.3f} m opening in 2F slab")

    # ---- step 6: railings ------------------------------------------------------ #
    def _railing(self, storey, name, pts_world, predefined):
        p0 = np.array(pts_world[0], float)
        rel = [tuple(np.array(p, float) - p0) for p in pts_world]
        r = self.new("IfcRailing", name, storey, mat4(p0), predefined=predefined)
        # Work around an IfcOpenShell 0.8.5-post1 bug: the railing helper's
        # default clear_width calls convert_si_to_unit(), whose Usecase.settings
        # is not initialized in this build. Supply the equivalent project-unit
        # value explicitly instead of triggering that broken default.
        import ifcopenshell.util.unit
        unit_scale = ifcopenshell.util.unit.calculate_unit_scale(self.f)
        clear_width_project_units = 0.04 / unit_scale
        self.set_rep(r, self.api.geometry.add_railing_representation(
            self.f, context=self.body, railing_path=rel, height=RAIL_H, railing_diameter=RAIL_DIA,
            support_spacing=RAIL_POST_SPACING, clear_width=clear_width_project_units,
            terminal_type="180"))
        return r

    def build_railings(self, stair):
        if self.f.by_type("IfcRailing"):
            warn("Railings already exist - skipping railing creation.")
            return
        info("Step 6: railings")
        gf, f2 = self.storeys["GF"], self.storeys["2F"]
        inset = 0.03
        n = 0
        if stair:
            T, rh, nt = TREAD_D, RISER_H, TREADS_PER_FLIGHT
            run = nt * T
            ya, yb = stair["y_f1"]
            yc, yd = stair["y_f2"]
            zm = stair["z_mid"]
            # flight 1 - south edge and well edge (top of rail = nosing + RAIL_H)
            for nm, y in (("Stair F1 south", ya + inset), ("Stair F1 well", yb - inset)):
                self._railing(gf, f"Railing {nm}",
                              [(STAIR_X0, y, rh + RAIL_H), (STAIR_X0 + run, y, (nt + 1) * rh + RAIL_H)], "HANDRAIL")
                n += 1
            # flight 2 - well edge and north edge (climbs west)
            for nm, y in (("Stair F2 well", yc + inset), ("Stair F2 north", yd - inset)):
                self._railing(gf, f"Railing {nm}",
                              [(stair["x_end"], y, zm + rh + RAIL_H),
                               (stair["x_end"] - run, y, zm + (nt + 1) * rh + RAIL_H)], "HANDRAIL")
                n += 1
            # landing south / north edges
            for nm, y in (("Landing south", STAIR_Y0 + inset), ("Landing north", STAIR_Y1 - inset)):
                self._railing(gf, f"Railing {nm}",
                              [(stair["x_end"], y, zm + RAIL_H), (EAST_WALL_INNER_X - 0.03, y, zm + RAIL_H)],
                              "GUARDRAIL")
                n += 1
        # 2F barrier round the stair opening (north edge + west edge; south = wall, east = wall)
        zt = ELEV_2F + RAIL_H
        self._railing(f2, "Railing Stair Opening 2F",
                      [(STAIR_X0 - inset, STAIR_Y0 + 0.05, zt), (STAIR_X0 - inset, STAIR_Y1 + inset, zt),
                       (EAST_WALL_INNER_X - 0.03, STAIR_Y1 + inset, zt)], "GUARDRAIL")
        # 2F balcony barrier: west edge then south edge (slab edge, inset)
        self._railing(f2, "Railing Balcony 2F",
                      [(-3.273 + inset, -1.63 - 0.05, zt), (-3.273 + inset, -3.100 + inset, zt),
                       (0.227 - 0.03, -3.100 + inset, zt)], "GUARDRAIL")
        ok(f"created {n + 2} railings")

    # ---- step 7: roof ---------------------------------------------------------- #
    def build_roof(self):
        if self.f.by_type("IfcRoof"):
            warn("An IfcRoof already exists - skipping roof creation.")
            return
        info("Step 7: roof")
        roof_st = self.storeys["Roof"]
        slab2f = next((e for e in contained(self.storeys["2F"]) if e.is_a("IfcSlab")), None)
        if slab2f is None:
            err("No 2F slab found - cannot derive roof outline")
            return
        # --- 7a: flat roof slab at wall-top level, same outline as the 2F slab
        body_item = next(r for r in slab2f.Representation.Representations
                         if r.RepresentationIdentifier == "Body").Items[0]
        poly = [tuple(p) for p in body_item.SweptArea.OuterCurve.Points.CoordList]
        if poly[0] == poly[-1]:
            poly = poly[:-1]
        sm = world(slab2f)
        slab_t = by_name(self.f, "IfcSlabType", "Slab Roof")
        rs = self.new("IfcSlab", "Roof Slab", roof_st, mat4((sm[0, 3], sm[1, 3], ELEV_ROOF)), predefined="ROOF")
        self.api.type.assign_type(self.f, related_objects=[rs], relating_type=slab_t)
        self.set_rep(rs, self.api.geometry.add_slab_representation(
            self.f, context=self.body, depth=0.1, direction_sense="POSITIVE", offset=-0.1, polyline=poly))
        try:   # mirror the existing 2F slab's layer-set usage offset
            usage = self.ifcopenshell.util.element.get_material(rs)
            if usage and usage.is_a("IfcMaterialLayerSetUsage"):
                self.api.material.edit_layer_usage(self.f, usage=usage,
                                                   attributes={"OffsetFromReferenceLine": -0.1})
        except Exception as e:
            warn(f"could not align roof slab layer usage offset ({e})")
        # --- 7b: hip roof over the 2F footprint (+ overhang)
        pts = np.array([sm @ np.array([p[0], p[1], 0.0, 1.0]) for p in poly])[:, :2]
        xmin, ymin = pts.min(0) - ROOF_OVERHANG
        xmax, ymax = pts.max(0) + ROOF_OVERHANG
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        hx, hy = (xmax - xmin) / 2, (ymax - ymin) / 2
        verts, faces, height = self._hip_mesh(hx, hy, ROOF_ANGLE, ROOF_THICK)
        roof = self.new("IfcRoof", "Roof", roof_st, mat4((cx, cy, ELEV_ROOF)), predefined="HIP_ROOF")
        rt = self.f.by_type("IfcRoofType")[0]
        self.api.type.assign_type(self.f, related_objects=[roof], relating_type=rt, should_map_representations=False)
        self.set_rep(roof, self.api.geometry.add_mesh_representation(
            self.f, context=self.body, vertices=[verts], faces=[faces]))
        # keep Bonsai's parametric data consistent with the mesh we generated
        import ifcopenshell.util.element as uel
        tdata = json.loads(uel.get_psets(rt)["BBIM_Roof"]["Data"])
        tdata["path_data"] = {"edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
                              "verts": [[-hx, -hy, 0.0], [-hx, hy, 0.0], [hx, hy, 0.0], [hx, -hy, 0.0]]}
        tdata["angle"], tdata["roof_thickness"] = ROOF_ANGLE, ROOF_THICK
        ps = self.api.pset.add_pset(self.f, product=roof, name="BBIM_Roof")
        self.api.pset.edit_pset(self.f, pset=ps, properties={"Data": json.dumps(tdata)})
        ok(f"roof: footprint {2 * hx:.2f} x {2 * hy:.2f} m, ridge rise {height:.2f} m, "
           f"overhang {ROOF_OVERHANG} m")

    @staticmethod
    def _hip_mesh(hx, hy, angle, thickness):
        """Closed hip-roof shell with vertical eave cut. Local origin = footprint centre, z=0 = eave."""
        if abs(hx - hy) < 1e-9:
            hy = hy + 1e-3                                    # avoid a degenerate ridge
        short = min(hx, hy)
        h = short * math.tan(angle)
        if hx < hy:
            r1, r2 = (0, -(hy - hx), h), (0, (hy - hx), h)
            top = {"S": [0, 1, 4], "E": [1, 2, 5, 4], "N": [2, 3, 5], "W": [3, 0, 4, 5]}
        else:
            r1, r2 = (-(hx - hy), 0, h), ((hx - hy), 0, h)
            top = {"S": [0, 1, 5, 4], "E": [1, 2, 5], "N": [2, 3, 4, 5], "W": [3, 0, 4]}
        c = [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)]
        t = c + [r1, r2]                                       # 0..3 eaves, 4..5 ridge
        d = thickness / math.cos(angle)
        b = [(x, y, z - d) for x, y, z in t]
        verts = t + b                                          # bottom index = +6
        faces = [f for f in top.values()]
        faces += [[i + 6 for i in reversed(f)] for f in top.values()]
        for i in range(4):                                     # vertical eave faces
            j = (i + 1) % 4
            faces.append([i + 6, j + 6, j, i])
        faces = outward(verts, faces)
        return [tuple(map(float, v)) for v in verts], [tuple(f) for f in faces], h


# --------------------------------------------------------------------------- #
#  VALIDATION
# --------------------------------------------------------------------------- #

def validate(path, source_path, pre_world, created_guids):
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element as uel
    info("---- validation ----")
    f = ifcopenshell.open(path)
    src = ifcopenshell.open(source_path)

    # spatial hierarchy ---------------------------------------------------------
    proj = f.by_type("IfcProject")
    if len(proj) != 1:
        err(f"expected 1 IfcProject, found {len(proj)}")
    chain_ok = True
    for s in f.by_type("IfcBuildingStorey"):
        b = uel.get_aggregate(s)
        if not b or not b.is_a("IfcBuilding"):
            chain_ok = False
    if chain_ok:
        ok("every storey aggregates into IfcBuilding -> IfcSite -> IfcProject")
    else:
        err("a storey is not aggregated into an IfcBuilding")
    elev = {s.Name: s.Elevation for s in f.by_type("IfcBuildingStorey")}
    info(f"storey elevations: {elev}")
    if elev.get("GF") != ELEV_GF or elev.get("2F") != ELEV_2F or elev.get("Roof") != ELEV_ROOF:
        err("storey elevations are not GF 0 / 2F 3 / Roof 6")

    # containment ---------------------------------------------------------------
    orphans = [e for e in f.by_type("IfcElement")
               if not e.is_a("IfcOpeningElement") and uel.get_container(e) is None]
    if orphans:
        err(f"{len(orphans)} element(s) are not contained in a storey: {[o.Name for o in orphans][:5]}")
    else:
        ok("every building element is contained in a storey")

    # counts per storey ------------------------------------------------------------
    for s in f.by_type("IfcBuildingStorey"):
        counts = {}
        for e in contained(s):
            counts[e.is_a()] = counts.get(e.is_a(), 0) + 1
        info(f"  {s.Name:4s}: {dict(sorted(counts.items()))}")
    st2 = next(s for s in f.by_type("IfcBuildingStorey") if s.Name == "2F")
    c2 = {}
    for e in contained(st2):
        c2[e.is_a()] = c2.get(e.is_a(), 0) + 1
    for cls, n in EXPECTED_2F.items():
        if c2.get(cls, 0) != n:
            err(f"2F {cls}: expected {n}, found {c2.get(cls, 0)}")
    if not any(e.is_a("IfcStairFlight") for e in f.by_type("IfcStairFlight")) or \
            len(f.by_type("IfcStairFlight")) < 2:
        err("fewer than 2 IfcStairFlight in the model")
    for cls in ("IfcRailing", "IfcRoof"):
        if not f.by_type(cls):
            err(f"no {cls} in the model")

    # openings / fillings ---------------------------------------------------------------
    bad = 0
    for e in f.by_type("IfcDoor") + f.by_type("IfcWindow"):
        if not e.FillsVoids or not e.FillsVoids[0].RelatingOpeningElement.VoidsElements:
            bad += 1
    for o in f.by_type("IfcOpeningElement"):
        if not o.VoidsElements:
            bad += 1
    if bad:
        err(f"{bad} door/window/opening relationship problem(s)")
    else:
        ok("every door/window fills an opening that voids a wall; every opening voids a host")

    # placement + representation ----------------------------------------------------------
    missing = [e for e in f.by_type("IfcElement") if not e.ObjectPlacement or not e.Representation]
    if missing:
        err(f"{len(missing)} element(s) lack placement or representation: {[m.Name for m in missing][:5]}")
    else:
        ok("all elements have an ObjectPlacement and a Representation")

    # original elements preserved -----------------------------------------------------------
    src_guids = {e.GlobalId for e in src.by_type("IfcRoot")}
    out_guids = {e.GlobalId for e in f.by_type("IfcRoot")}
    lost = src_guids - out_guids
    if lost:
        err(f"{len(lost)} original GlobalIds are missing from the output")
    else:
        ok(f"all {len(src_guids)} original IfcRoot objects preserved (same GlobalIds)")
    shifted = 0
    for gid, m in pre_world.items():
        try:
            if not np.allclose(m, world(f.by_guid(gid)), atol=1e-6):
                shifted += 1
        except Exception:
            shifted += 1
    if shifted:
        err(f"{shifted} original element(s) changed world position")
    else:
        ok("no original element changed world position")

    # geometry + duplicate detection ----------------------------------------------------------
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    boxes, failed = {}, []
    for e in f.by_type("IfcElement"):
        if e.is_a("IfcOpeningElement"):
            continue
        try:
            sh = ifcopenshell.geom.create_shape(s, e)
            v = np.array(sh.geometry.verts).reshape(-1, 3)
            if len(v):
                boxes[e.GlobalId] = (e, v.min(0), v.max(0))
        except Exception:
            failed.append(e)
    new_fail = [e for e in failed if e.GlobalId in created_guids]
    old_fail = [e for e in failed if e.GlobalId not in created_guids]
    if new_fail:
        warn(f"{len(new_fail)} NEW element(s) did not tessellate headless: {[e.Name for e in new_fail][:6]}")
    else:
        ok("all new elements tessellate (geometry is valid)")
    if old_fail:
        info(f"  ({len(old_fail)} ORIGINAL doors/windows have Profile representations that the "
             f"headless tessellator rejects - they are unchanged from your file)")
    dup = []
    items = list(boxes.values())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a[0].is_a() == b[0].is_a() and np.allclose(a[1], b[1], atol=1e-3) and np.allclose(a[2], b[2], atol=1e-3):
                dup.append((a[0].Name, b[0].Name))
    if dup:
        err(f"duplicate elements (identical class and bounding box): {dup[:6]}")
    else:
        ok("no duplicate elements (class + bounding box)")

    # collisions between new walls (overlap volume) -------------------------------------------------
    nw = [v for v in boxes.values() if v[0].is_a("IfcWall") and v[0].GlobalId in created_guids]
    ov = []
    for i in range(len(nw)):
        for j in range(i + 1, len(nw)):
            lo = np.maximum(nw[i][1], nw[j][1]); hi = np.minimum(nw[i][2], nw[j][2])
            d = hi - lo
            if np.all(d > 1e-3):
                ov.append((nw[i][0].Name, nw[j][0].Name, round(float(np.prod(d)), 4)))
    if ov:
        warn(f"new walls with overlapping volume: {ov[:6]}")
    else:
        ok("new walls butt-join with no overlapping volumes")

    # schema validation ---------------------------------------------------------------------------------
    try:
        import ifcopenshell.validate
        lg = ifcopenshell.validate.json_logger()
        ifcopenshell.validate.validate(f, lg)
        new_i = len(lg.statements)
        lg0 = ifcopenshell.validate.json_logger()
        ifcopenshell.validate.validate(src, lg0)
        old_i = len(lg0.statements)
        (ok if new_i <= old_i else warn)(f"ifcopenshell.validate: {new_i} issue(s) in output vs {old_i} in your source file")
        if new_i > old_i:
            seen = set()
            for st in lg.statements:
                k = st.get("message", "")[:140]
                if k not in seen and len(seen) < 6:
                    seen.add(k); info(f"    {k}")
    except Exception as e:
        warn(f"schema validation skipped ({e})")
    return f


# --------------------------------------------------------------------------- #
#  BLENDER / BONSAI EXPORT  (only when running inside Blender)
# --------------------------------------------------------------------------- #

def save_blend(ifc_path, blend_path):
    if not in_blender():
        warn("Not running inside Blender -> s03-s04.blend NOT created. Run the script from Blender's "
             "Scripting workspace (or `blender --background --python ...`) to also get the .blend.")
        return False
    import bpy
    try:
        try:
            bpy.ops.bim.load_project(filepath=ifc_path, should_start_fresh_load=True)
        except TypeError:
            bpy.ops.bim.load_project(filepath=ifc_path)
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        ok(f"saved Blender file: {blend_path}")
        return True
    except Exception as e:
        warn(f"Could not load the IFC into Bonsai / save .blend automatically ({e}). Manual fallback: "
             f"BIM tab -> Project Overview -> Load Project -> {ifc_path}; then File > Save As -> s03-s04.blend")
        return False


# --------------------------------------------------------------------------- #
#  MAIN
# --------------------------------------------------------------------------- #

def main():
    check_environment()
    opts = resolve_args()
    src = opts["source"]
    os.makedirs(opts["out_dir"], exist_ok=True)
    out_ifc = os.path.join(opts["out_dir"], OUTPUT_IFC_NAME)
    out_blend = os.path.join(opts["out_dir"], OUTPUT_BLEND_NAME)
    if os.path.normcase(os.path.abspath(out_ifc)) == os.path.normcase(src):
        err("Output path equals the source path - refusing to overwrite your original.")
        raise SystemExit(1)

    info(f"source : {src}")
    info(f"output : {out_ifc}")
    shutil.copy2(src, out_ifc)                    # ALL edits happen on the copy
    ok("working copy created (original is untouched)")

    b = Builder(out_ifc)
    pre_world = b.fix_storeys()
    b.build_walls()
    b.build_doors_windows()
    stair = b.build_stair()
    b.build_stair_opening()
    b.build_railings(stair)
    b.build_roof()

    b.f.write(out_ifc)
    ok(f"wrote {out_ifc}")

    created_guids = {e.GlobalId for e in b.created}
    validate(out_ifc, src, pre_world, created_guids)

    if opts["blender"]:
        save_blend(out_ifc, out_blend)

    info("---- summary ----")
    info(f"created {len(b.created)} new IFC objects; {len(_WARNINGS)} warning(s), {len(_ERRORS)} error(s)")
    for w in _WARNINGS:
        print(f"   warn: {w}")
    for e in _ERRORS:
        print(f"   FAIL: {e}")
    return 1 if _ERRORS else 0


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as e:
        code = e.code or 0
    except Exception:
        traceback.print_exc()
        err("unhandled exception - see traceback above (your source file was not modified)")
        code = 2
    if not in_blender():
        sys.exit(code)

# --------------------------------------------------------------------------- #
# README  -  how to run
# --------------------------------------------------------------------------- #
# A) INSIDE BLENDER 4.5 + Bonsai 0.8.x  (gives you the .blend too)
#    1. Put s03-discussion.ifc somewhere (default expected: D:\s03-discussion.ifc).
#    2. Blender -> Scripting workspace -> Open -> s03_s04_complete.py
#       (if the file is elsewhere, change SOURCE_CANDIDATES at the top).
#    3. Window -> Toggle System Console (Windows) to watch progress, then Run Script (Alt+P).
#    4. Results are written NEXT TO the source: s03-s04.ifc and s03-s04.blend.
#
# B) BLENDER, NO GUI
#    blender --background --python s03_s04_complete.py -- --source "D:/s03-discussion.ifc"
#
# C) PLAIN PYTHON  (IFC only)
#    pip install ifcopenshell
#    python s03_s04_complete.py --source D:/s03-discussion.ifc --out-dir D:/out
#
# Flags: --source PATH   --out-dir DIR   --no-blender
