
bl_info = {
    "name": "GLB Preflight",
    "author": "Akinaru Iino / IINO Lab.",
    "version": (0, 1, 4),
    "blender": (4, 5, 0),
    "location": "3D View > Sidebar > GLB Preflight",
    "description": "Check and prepare Blender models for reliable GLB/glTF export",
    "category": "Import-Export",
}

import bpy
import bmesh
import json
import statistics
import math
import os
from mathutils import Vector
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import BoolProperty, FloatProperty, StringProperty, PointerProperty

RESULT_KEY = "glb_preflight_last_result"
REPAIR_COLLECTION_NAME = "GLB_READY"

def mesh_objects_from_context(context, selected_only=False):
    return [o for o in (context.selected_objects if selected_only else context.scene.objects)
            if o.type == 'MESH' and REPAIR_COLLECTION_NAME not in [c.name for c in o.users_collection]]


def count_mesh_issues(obj, eps=1e-10):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()

    face_count = len(bm.faces)
    edge_count = len(bm.edges)
    vert_count = len(bm.verts)

    if face_count == 0 and edge_count > 0:
        geometry_type = "LINE"
    elif face_count > 0:
        geometry_type = "SURFACE"
    elif vert_count > 0:
        geometry_type = "POINT"
    else:
        geometry_type = "EMPTY"

    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    boundary = sum(1 for e in bm.edges if e.is_boundary)
    loose_edges = sum(1 for e in bm.edges if e.is_wire)
    loose_verts = sum(1 for v in bm.verts if len(v.link_edges) == 0)
    ngons = sum(1 for f in bm.faces if len(f.verts) > 4)
    degenerate = sum(1 for f in bm.faces if f.calc_area() <= eps)

    flipped = 0
    if geometry_type == "SURFACE" and bm.verts and bm.faces:
        center = sum((v.co for v in bm.verts), Vector()) / len(bm.verts)
        for f in bm.faces:
            d = f.calc_center_median() - center
            if d.length > 1e-12 and f.normal.dot(d.normalized()) < -0.5:
                flipped += 1

    bm.free()

    sx, sy, sz = obj.scale
    return {
        "name": obj.name,
        "geometry_type": geometry_type,
        "vertices": len(obj.data.vertices),
        "edges": len(obj.data.edges),
        "faces": len(obj.data.polygons),
        "triangles_est": sum(max(1, len(p.vertices)-2) for p in obj.data.polygons),
        "non_manifold_edges": non_manifold,
        "boundary_edges": boundary,
        "loose_edges": loose_edges,
        "loose_verts": loose_verts,
        "degenerate_faces": degenerate,
        "ngons": ngons,
        "flipped_suspects": flipped,
        "negative_scale": (sx*sy*sz) < 0,
        "unapplied_scale": any(abs(v-1.0) > 1e-6 for v in obj.scale),
        "unapplied_rotation": any(abs(v) > 1e-6 for v in obj.rotation_euler),
        "missing_materials": sum(1 for s in obj.material_slots if s.material is None),
    }


def summarize(results):
    t = {k:0 for k in [
        "objects","surface_objects","line_objects","point_objects","empty_objects",
        "vertices","faces","triangles_est","non_manifold_edges",
        "boundary_edges","loose_edges","loose_verts","degenerate_faces","ngons",
        "flipped_suspects","negative_scale_objects","unapplied_scale_objects",
        "unapplied_rotation_objects","missing_material_slots"
    ]}
    t["objects"] = len(results)

    hard = 0
    warn = 0
    t["wire_only_objects"] = 0
    t["invalid_edge_objects"] = 0

    for r in results:
        gt = r["geometry_type"]
        t["surface_objects"] += int(gt == "SURFACE")
        t["line_objects"] += int(gt == "LINE")
        t["point_objects"] += int(gt == "POINT")
        t["empty_objects"] += int(gt == "EMPTY")

        for k in ["vertices","faces","triangles_est","non_manifold_edges","boundary_edges",
                  "loose_edges","loose_verts","degenerate_faces","ngons","flipped_suspects"]:
            t[k] += r[k]

        t["negative_scale_objects"] += int(r["negative_scale"])
        t["unapplied_scale_objects"] += int(r["unapplied_scale"])
        t["unapplied_rotation_objects"] += int(r["unapplied_rotation"])
        t["missing_material_slots"] += r["missing_materials"]

        # Classification-aware severity:
        # LINE objects may intentionally consist only of loose/wire edges.
        # SURFACE objects may legitimately have open boundaries (terrain, sheets).
        if gt == "SURFACE":
            # Open terrain surfaces are valid. Degenerate faces may cause export problems.
            hard += r["degenerate_faces"]
            warn += r["loose_verts"]
            warn += int(r["negative_scale"])
            warn += r["non_manifold_edges"]
            warn += r["ngons"]
            warn += r["missing_materials"]
            warn += int(r["unapplied_scale"])
            warn += int(r["unapplied_rotation"])
        elif gt == "LINE":
            t["wire_only_objects"] += 1
            # glTF does not represent arbitrary Blender wire-only meshes by default.
            warn += r["loose_verts"]
            warn += int(r["negative_scale"])
            warn += int(r["unapplied_scale"])
            warn += int(r["unapplied_rotation"])
        elif gt == "POINT":
            warn += r["loose_verts"]
        else:
            warn += 1

    t["status"] = "ERROR" if hard else ("WARNING" if warn else "PASS")
    t["hard_issue_count"] = hard
    t["warning_issue_count"] = warn

    # Web delivery size heuristic (informational only)
    tri = t["triangles_est"]
    if tri < 500_000:
        t["web_size_class"] = "LIGHT"
    elif tri < 2_000_000:
        t["web_size_class"] = "MEDIUM"
    elif tri < 5_000_000:
        t["web_size_class"] = "HEAVY"
    else:
        t["web_size_class"] = "VERY HEAVY"

    return t


def summary_text(t):
    return (
        f"Status: {t['status']}\n"
        f"Objects: {t['objects']}  Surface:{t['surface_objects']}  Line:{t['line_objects']}\n"
        f"Vertices: {t['vertices']:,}\n"
        f"Faces: {t['faces']:,}\n"
        f"Est. triangles: {t['triangles_est']:,} ({t['web_size_class']})\n"
        f"Non-manifold: {t['non_manifold_edges']:,}\n"
        f"Boundary edges: {t['boundary_edges']:,}\n"
        f"Loose edges: {t['loose_edges']:,} (line meshes may be normal)\n"
        f"Loose verts: {t['loose_verts']:,}\n"
        f"Degenerate faces: {t['degenerate_faces']:,}\n"
        f"N-gons: {t['ngons']:,}\n"
        f"Wire-only objects (GLB skipped): {t['wire_only_objects']:,}\n"
    )

def ensure_repair_collection(scene):
    old = bpy.data.collections.get(REPAIR_COLLECTION_NAME)
    if old:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)
    col = bpy.data.collections.new(REPAIR_COLLECTION_NAME)
    scene.collection.children.link(col)
    return col


def repair_mesh(obj, merge_distance=0.0001, triangulate=True, recalc_normals=True):
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    has_faces = len(bm.faces) > 0

    # Only treat wire edges as disposable when the object is a surface mesh.
    # For line-only meshes (e.g. roads/CAD linework), wire edges are the actual content.
    if has_faces:
        loose_edges = [e for e in bm.edges if e.is_wire]
        if loose_edges:
            bmesh.ops.delete(bm, geom=loose_edges, context='EDGES')

    loose_verts = [v for v in bm.verts if len(v.link_edges) == 0]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')

    if merge_distance > 0 and bm.verts:
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=merge_distance)

    if has_faces:
        try:
            bmesh.ops.dissolve_degenerate(
                bm,
                dist=max(merge_distance*0.1, 1e-9),
                edges=bm.edges[:]
            )
        except Exception:
            pass

        if recalc_normals and bm.faces:
            try:
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
            except Exception:
                pass

        if triangulate and bm.faces:
            try:
                bmesh.ops.triangulate(
                    bm,
                    faces=bm.faces[:],
                    quad_method='BEAUTY',
                    ngon_method='BEAUTY'
                )
            except Exception:
                pass

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def _sample_edge_lengths_world(obj, max_samples=50000):
    """Return a sampled list of edge lengths in world-space."""
    me = obj.data
    total = len(me.edges)
    if total == 0:
        return []
    step = max(1, total // max_samples)
    mw = obj.matrix_world
    out = []
    for i in range(0, total, step):
        e = me.edges[i]
        a = mw @ me.vertices[e.vertices[0]].co
        b = mw @ me.vertices[e.vertices[1]].co
        out.append((a - b).length)
        if len(out) >= max_samples:
            break
    return out



def analyze_active_object_geometry(
    obj,
    long_edge_factor=10.0,
    terrain_spike_factor=8.0,
    steep_face_angle=60.0,
):
    """Detailed active-mesh diagnostics.

    Adds two topology/terrain-oriented checks:
      - internal boundary loops (possible holes)
      - unusually steep surface faces

    These are heuristics. They identify places to inspect; they do not prove
    that the geometry is invalid.
    """
    if not obj or obj.type != 'MESH':
        return None

    me = obj.data
    sampled = _sample_edge_lengths_world(obj)
    median_len = statistics.median(sampled) if sampled else 0.0
    long_threshold = median_len * long_edge_factor if median_len > 0 else 0.0
    z_threshold = median_len * terrain_spike_factor if median_len > 0 else 0.0

    mw = obj.matrix_world
    long_edges = []
    steep_edges = []

    for e in me.edges:
        va = mw @ me.vertices[e.vertices[0]].co
        vb = mw @ me.vertices[e.vertices[1]].co
        length = (va - vb).length
        if long_threshold > 0 and length > long_threshold:
            long_edges.append(e.index)
        if z_threshold > 0:
            dz = abs(va.z - vb.z)
            horizontal = ((va.x - vb.x)**2 + (va.y - vb.y)**2) ** 0.5
            if horizontal > median_len * 0.25 and dz > z_threshold:
                steep_edges.append(e.index)

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()

    boundary_edges = [e.index for e in bm.edges if e.is_boundary]
    # "Internal non-manifold" excludes ordinary one-face boundaries.
    internal_non_manifold = [
        e.index for e in bm.edges
        if (not e.is_manifold) and (not e.is_boundary) and (not e.is_wire)
    ]
    degenerate_faces = [f.index for f in bm.faces if f.calc_area() <= 1e-10]
    ngons = [f.index for f in bm.faces if len(f.verts) > 4]

    # Build connected components of boundary edges.
    boundary_set = set(boundary_edges)
    vertex_to_edges = {}
    for ei in boundary_edges:
        e = bm.edges[ei]
        for v in e.verts:
            vertex_to_edges.setdefault(v.index, set()).add(ei)

    components = []
    unvisited = set(boundary_edges)
    while unvisited:
        seed = unvisited.pop()
        comp = {seed}
        stack = [seed]
        while stack:
            ei = stack.pop()
            e = bm.edges[ei]
            for v in e.verts:
                for nei in vertex_to_edges.get(v.index, ()):
                    if nei in unvisited:
                        unvisited.remove(nei)
                        comp.add(nei)
                        stack.append(nei)
        components.append(sorted(comp))

    # For a single connected terrain sheet, the largest boundary component is
    # normally the outer perimeter. Smaller components are likely internal holes.
    outer_boundary = []
    internal_hole_edges = []
    boundary_loop_count = len(components)
    if components:
        components.sort(key=len, reverse=True)
        outer_boundary = components[0]
        for comp in components[1:]:
            internal_hole_edges.extend(comp)

    # Face slope in world coordinates. 0 deg = horizontal, 90 deg = vertical.
    steep_faces = []
    try:
        normal_matrix = mw.to_3x3().inverted().transposed()
    except Exception:
        normal_matrix = mw.to_3x3()

    for f in bm.faces:
        wn = normal_matrix @ f.normal
        if wn.length <= 1e-12:
            continue
        wn.normalize()
        zc = max(-1.0, min(1.0, abs(wn.z)))
        slope = math.degrees(math.acos(zc))
        if slope >= steep_face_angle:
            steep_faces.append(f.index)

    bm.free()

    return {
        "object": obj.name,
        "median_edge_length_world": median_len,
        "long_edge_threshold": long_threshold,
        "terrain_spike_z_threshold": z_threshold,
        "steep_face_angle": steep_face_angle,
        "long_edges": long_edges,
        "terrain_spike_edges": steep_edges,
        "internal_non_manifold_edges": internal_non_manifold,
        "boundary_edges": boundary_edges,
        "boundary_loop_count": boundary_loop_count,
        "outer_boundary_edges": outer_boundary,
        "internal_hole_edges": sorted(set(internal_hole_edges)),
        "steep_faces": steep_faces,
        "degenerate_faces": degenerate_faces,
        "ngon_faces": ngons,
    }

def _activate_object_edit_mode(context, obj):
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')


def select_mesh_indices(context, obj, domain, indices):
    """Select vertices/edges/faces by mesh index and enter Edit Mode."""
    if not obj or obj.type != 'MESH':
        return False
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    me = obj.data
    for v in me.vertices:
        v.select = False
    for e in me.edges:
        e.select = False
    for p in me.polygons:
        p.select = False

    if domain == 'EDGE':
        for i in indices:
            if 0 <= i < len(me.edges):
                me.edges[i].select = True
    elif domain == 'FACE':
        for i in indices:
            if 0 <= i < len(me.polygons):
                me.polygons[i].select = True
    elif domain == 'VERT':
        for i in indices:
            if 0 <= i < len(me.vertices):
                me.vertices[i].select = True

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    context.view_layer.objects.active = obj

    if domain == 'EDGE':
        context.tool_settings.mesh_select_mode = (False, True, False)
    elif domain == 'FACE':
        context.tool_settings.mesh_select_mode = (False, False, True)
    else:
        context.tool_settings.mesh_select_mode = (True, False, False)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Re-apply selections after mode switch because Blender may flush them.
    if domain == 'EDGE':
        for i in indices:
            if 0 <= i < len(me.edges):
                me.edges[i].select = True
    elif domain == 'FACE':
        for i in indices:
            if 0 <= i < len(me.polygons):
                me.polygons[i].select = True
    else:
        for i in indices:
            if 0 <= i < len(me.vertices):
                me.vertices[i].select = True

    bpy.ops.object.mode_set(mode='EDIT')
    return True


class GLBPreflightSettings(PropertyGroup):
    selected_only: BoolProperty(name="Selected only", default=False)
    merge_distance: FloatProperty(name="Merge distance", default=0.0, min=0.0, soft_max=0.01, precision=6)
    triangulate: BoolProperty(name="Triangulate", default=False)
    recalc_normals: BoolProperty(name="Recalculate normals", default=False)
    apply_transforms: BoolProperty(name="Bake world transforms into copy", default=False)
    safe_geometry: BoolProperty(
        name="Conservative mesh cleanup", default=False,
        description="Optional: cleanup on copies; leave off first for fragile terrain meshes")
    include_wire: BoolProperty(
        name="Include wire-only objects (not GLB visible)", default=False,
        description="Wire-only mesh objects normally have no rendered triangles in GLB")
    long_edge_factor: FloatProperty(
        name="Long edge factor",
        default=10.0,
        min=2.0,
        soft_max=50.0,
        description="Flag edges longer than this multiple of the sampled median edge length"
    )
    terrain_spike_factor: FloatProperty(
        name="Terrain spike factor",
        default=8.0,
        min=2.0,
        soft_max=50.0,
        description="Terrain-oriented heuristic: flag edges with unusually large Z differences"
    )
    steep_face_angle: FloatProperty(
        name="Steep face angle (deg)",
        default=60.0,
        min=10.0,
        max=89.9,
        description="Flag surface faces steeper than this angle in degrees"
    )
    export_path: StringProperty(name="GLB file", subtype='FILE_PATH', default="//GLB_READY.glb")

class GLB_PREFLIGHT_OT_check(Operator):
    bl_idname = "glb_preflight.check"
    bl_label = "Run Preflight Check"
    def execute(self, context):
        s = context.scene.glb_preflight_settings
        objs = mesh_objects_from_context(context, s.selected_only)
        if not objs:
            self.report({'WARNING'}, "No mesh objects found.")
            return {'CANCELLED'}
        results = [count_mesh_issues(o) for o in objs]
        totals = summarize(results)
        context.scene[RESULT_KEY] = json.dumps({"totals":totals,"objects":results}, ensure_ascii=False)
        context.scene["glb_preflight_status"] = totals["status"]
        context.scene["glb_preflight_summary"] = summary_text(totals)
        self.report({'INFO'}, f"GLB Preflight: {totals['status']}")
        return {'FINISHED'}


class GLB_PREFLIGHT_OT_analyze_active(Operator):
    bl_idname = "glb_preflight.analyze_active"
    bl_label = "Analyze Active Object"
    bl_description = "Run detailed diagnostics on the active mesh object"

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Select one mesh object first.")
            return {'CANCELLED'}

        s = context.scene.glb_preflight_settings
        result = analyze_active_object_geometry(
            obj,
            long_edge_factor=s.long_edge_factor,
            terrain_spike_factor=s.terrain_spike_factor,
            steep_face_angle=s.steep_face_angle,
        )
        context.scene["glb_preflight_active_analysis"] = json.dumps(result, ensure_ascii=False)
        context.scene["glb_preflight_active_summary"] = (
            f"{result['object']} | "
            f"Long:{len(result['long_edges'])} "
            f"Spike:{len(result['terrain_spike_edges'])} "
            f"Holes:{max(0, result['boundary_loop_count'] - 1)} "
            f"SteepFaces:{len(result['steep_faces'])} "
            f"InternalNonManifold:{len(result['internal_non_manifold_edges'])}"
        )
        self.report({'INFO'}, f"Analyzed {obj.name}")
        return {'FINISHED'}


class GLB_PREFLIGHT_OT_select_issue(Operator):
    bl_idname = "glb_preflight.select_issue"
    bl_label = "Select Issue"
    bl_description = "Select the detected issue elements on the active object"

    issue_type: bpy.props.EnumProperty(
        items=[
            ('LONG', "Long edges", ""),
            ('SPIKE', "Terrain spike edges", ""),
            ('NONMANIFOLD', "Internal non-manifold edges", ""),
            ('BOUNDARY', "Boundary edges", ""),
            ('HOLE', "Internal hole edges", ""),
            ('STEEP', "Steep faces", ""),
            ('DEGENERATE', "Degenerate faces", ""),
            ('NGON', "N-gon faces", ""),
        ]
    )

    def execute(self, context):
        obj = context.active_object
        raw = context.scene.get("glb_preflight_active_analysis", "")
        if not obj or obj.type != 'MESH' or not raw:
            self.report({'WARNING'}, "Run Analyze Active Object first.")
            return {'CANCELLED'}

        try:
            data = json.loads(raw)
        except Exception:
            self.report({'ERROR'}, "Could not read active-object analysis.")
            return {'CANCELLED'}

        if data.get("object") != obj.name:
            self.report({'WARNING'}, "Active object changed. Analyze it again first.")
            return {'CANCELLED'}

        mapping = {
            'LONG': ('EDGE', data.get('long_edges', [])),
            'SPIKE': ('EDGE', data.get('terrain_spike_edges', [])),
            'NONMANIFOLD': ('EDGE', data.get('internal_non_manifold_edges', [])),
            'BOUNDARY': ('EDGE', data.get('boundary_edges', [])),
            'HOLE': ('EDGE', data.get('internal_hole_edges', [])),
            'STEEP': ('FACE', data.get('steep_faces', [])),
            'DEGENERATE': ('FACE', data.get('degenerate_faces', [])),
            'NGON': ('FACE', data.get('ngon_faces', [])),
        }
        domain, indices = mapping[self.issue_type]

        if not indices:
            self.report({'INFO'}, "No matching issues detected.")
            return {'CANCELLED'}

        if not select_mesh_indices(context, obj, domain, indices):
            self.report({'ERROR'}, "Selection failed.")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Selected {len(indices)} issue elements.")
        return {'FINISHED'}


def ready_mesh_objects(col):
    return [o for o in col.all_objects if o.type == 'MESH' and len(o.data.polygons) > 0]


class GLB_PREFLIGHT_OT_repair(Operator):
    bl_idname = "glb_preflight.repair"
    bl_label = "Create GLB_READY Copy"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.mode != 'OBJECT':
            self.report({'WARNING'}, "Switch to Object Mode before creating the GLB copy.")
            return {'CANCELLED'}
        s = context.scene.glb_preflight_settings
        src = [o for o in mesh_objects_from_context(context, s.selected_only)
               if REPAIR_COLLECTION_NAME not in [c.name for c in o.users_collection]]
        if not src:
            self.report({'WARNING'}, "No source meshes found. Select original objects and retry.")
            return {'CANCELLED'}
        surfaces = [o for o in src if len(o.data.polygons) > 0]
        wires = [o for o in src if len(o.data.polygons) == 0]
        if not surfaces:
            self.report({'ERROR'}, "No surface meshes to export. GLB requires triangles.")
            return {'CANCELLED'}

        # Keep original source collection until all copies are ready. Originals are untouched.
        new_col = bpy.data.collections.new(REPAIR_COLLECTION_NAME + "_new")
        context.scene.collection.children.link(new_col)
        created = []
        try:
            for obj in src:
                if len(obj.data.polygons) == 0 and not s.include_wire:
                    continue
                cp = obj.copy()
                cp.data = obj.data.copy()
                cp.animation_data_clear()
                # Avoid inherited parent transforms becoming incorrect when exported alone.
                world = obj.matrix_world.copy()
                cp.parent = None
                new_col.objects.link(cp)
                cp.matrix_world = world
                if s.apply_transforms:
                    cp.data.transform(cp.matrix_world)
                    cp.matrix_world.identity()
                if s.safe_geometry:
                    repair_mesh(cp, s.merge_distance, s.triangulate, s.recalc_normals)
                cp["glb_preflight_source"] = obj.name
                created.append(cp)
        except Exception as exc:
            for o in list(new_col.objects):
                bpy.data.objects.remove(o, do_unlink=True)
            bpy.data.collections.remove(new_col)
            self.report({'ERROR'}, f"Copy failed: {str(exc)[:160]}")
            return {'CANCELLED'}

        old = bpy.data.collections.get(REPAIR_COLLECTION_NAME)
        if old:
            for o in list(old.objects):
                bpy.data.objects.remove(o, do_unlink=True)
            bpy.data.collections.remove(old)
        new_col.name = REPAIR_COLLECTION_NAME
        for o in context.selected_objects:
            o.select_set(False)
        for o in created:
            o.select_set(True)
        context.view_layer.objects.active = created[0]
        tris = sum(sum(max(1, len(p.vertices)-2) for p in o.data.polygons) for o in created)
        context.scene["glb_preflight_ready_summary"] = (
            f"Ready: {len(created)} objects, ~{tris:,} triangles. "
            f"Skipped {len(wires) if not s.include_wire else 0} wire-only meshes.")
        self.report({'INFO'}, f"GLB_READY created; {len(wires) if not s.include_wire else 0} wire meshes skipped.")
        return {'FINISHED'}


class GLB_PREFLIGHT_OT_export(Operator):
    bl_idname = "glb_preflight.export_glb"
    bl_label = "Export GLB_READY"

    def execute(self, context):
        if context.mode != 'OBJECT':
            self.report({'WARNING'}, "Switch to Object Mode before exporting.")
            return {'CANCELLED'}
        s = context.scene.glb_preflight_settings
        col = bpy.data.collections.get(REPAIR_COLLECTION_NAME)
        if not col:
            self.report({'ERROR'}, "GLB_READY not found. Create the copy first.")
            return {'CANCELLED'}
        objs = ready_mesh_objects(col)
        if not objs:
            self.report({'ERROR'}, "No surface meshes in GLB_READY.")
            return {'CANCELLED'}
        if not hasattr(bpy.ops.export_scene, 'gltf'):
            self.report({'ERROR'}, "Blender glTF 2.0 export add-on is not available.")
            return {'CANCELLED'}
        if s.export_path.startswith('//') and not bpy.data.filepath:
            self.report({'ERROR'}, "Save the .blend first, or specify an absolute GLB file path.")
            return {'CANCELLED'}
        path = bpy.path.abspath(s.export_path)
        if not path.lower().endswith('.glb'):
            path += '.glb'
        folder = os.path.dirname(path)
        if not os.path.isdir(folder):
            self.report({'ERROR'}, "Output folder does not exist. Choose an existing folder.")
            return {'CANCELLED'}

        selected_before = list(context.selected_objects)
        active_before = context.view_layer.objects.active
        try:
            for o in selected_before:
                o.select_set(False)
            for o in objs:
                o.hide_set(False)
                o.select_set(True)
            context.view_layer.objects.active = objs[0]
            # Use Blender's official glTF exporter; do NOT write a custom GLB.
            result = bpy.ops.export_scene.gltf(
                filepath=path, export_format='GLB', use_selection=True,
                export_apply=False,
            )
            if 'FINISHED' not in result:
                raise RuntimeError(f"Exporter returned {result}")
        except Exception as exc:
            self.report({'ERROR'}, f"GLB export failed: {str(exc)[:180]}")
            return {'CANCELLED'}
        finally:
            for o in objs:
                o.select_set(False)
            for o in selected_before:
                if o.name in bpy.data.objects:
                    o.select_set(True)
            if active_before and active_before.name in bpy.data.objects:
                context.view_layer.objects.active = active_before
        self.report({'INFO'}, f"GLB exported: {path}")
        return {'FINISHED'}


class GLB_PREFLIGHT_OT_clear(Operator):
    bl_idname = "glb_preflight.clear"
    bl_label = "Remove GLB_READY"
    bl_options = {'REGISTER','UNDO'}
    def execute(self, context):
        col = bpy.data.collections.get(REPAIR_COLLECTION_NAME)
        if col:
            for obj in list(col.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(col)
        return {'FINISHED'}

class GLB_PREFLIGHT_PT_panel(Panel):
    bl_label = "GLB Preflight"
    bl_idname = "GLB_PREFLIGHT_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GLB Preflight'
    def draw(self, context):
        layout = self.layout
        s = context.scene.glb_preflight_settings

        box = layout.box()
        box.label(text="1. Check")
        box.prop(s, "selected_only")
        box.operator("glb_preflight.check", icon='CHECKMARK')
        box.label(text=f"Status: {context.scene.get('glb_preflight_status','NOT CHECKED')}")
        summary = context.scene.get("glb_preflight_summary","")
        for line in summary.splitlines()[:8]:
            box.label(text=line)

        box = layout.box()
        box.label(text="2. Locate issues")
        box.prop(s, "long_edge_factor")
        box.prop(s, "terrain_spike_factor")
        box.prop(s, "steep_face_angle")
        box.operator("glb_preflight.analyze_active", icon='VIEWZOOM')
        active_summary = context.scene.get("glb_preflight_active_summary", "")
        if active_summary:
            box.label(text=active_summary)

        row = box.row(align=True)
        op = row.operator("glb_preflight.select_issue", text="Long edges")
        op.issue_type = 'LONG'
        op = row.operator("glb_preflight.select_issue", text="Z spikes")
        op.issue_type = 'SPIKE'

        row = box.row(align=True)
        op = row.operator("glb_preflight.select_issue", text="Internal NM")
        op.issue_type = 'NONMANIFOLD'
        op = row.operator("glb_preflight.select_issue", text="Boundary")
        op.issue_type = 'BOUNDARY'

        row = box.row(align=True)
        op = row.operator("glb_preflight.select_issue", text="Internal holes")
        op.issue_type = 'HOLE'
        op = row.operator("glb_preflight.select_issue", text="Steep faces")
        op.issue_type = 'STEEP'

        row = box.row(align=True)
        op = row.operator("glb_preflight.select_issue", text="Degenerate")
        op.issue_type = 'DEGENERATE'
        op = row.operator("glb_preflight.select_issue", text="N-gons")
        op.issue_type = 'NGON'

        box = layout.box()
        box.label(text="3. Repair copy")
        box.label(text="Safe default: copy only, no topology changes")
        box.prop(s, "include_wire")
        box.prop(s, "safe_geometry")
        if s.safe_geometry:
            box.prop(s, "merge_distance")
            box.prop(s, "triangulate")
            box.prop(s, "recalc_normals")
        box.prop(s, "apply_transforms")
        box.operator("glb_preflight.repair", icon='MODIFIER')
        box.operator("glb_preflight.clear", icon='TRASH')
        box.label(text=context.scene.get("glb_preflight_ready_summary", ""))

        box = layout.box()
        box.label(text="4. Export")
        box.prop(s, "export_path")
        box.operator("glb_preflight.export_glb", icon='EXPORT')
        box.label(text="GLB may be very large; check in target viewer")

        layout.label(text="Original objects are not modified.")
        layout.label(text="Repair output: GLB_READY collection")

classes = (
    GLBPreflightSettings,
    GLB_PREFLIGHT_OT_check,
    GLB_PREFLIGHT_OT_analyze_active,
    GLB_PREFLIGHT_OT_select_issue,
    GLB_PREFLIGHT_OT_repair,
    GLB_PREFLIGHT_OT_export,
    GLB_PREFLIGHT_OT_clear,
    GLB_PREFLIGHT_PT_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.glb_preflight_settings = PointerProperty(type=GLBPreflightSettings)

def unregister():
    if hasattr(bpy.types.Scene, "glb_preflight_settings"):
        del bpy.types.Scene.glb_preflight_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
