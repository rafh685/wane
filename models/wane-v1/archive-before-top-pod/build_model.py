"""Wane exterior styling study. Run with Blender --background --python this_file.
No functional internal components or fabrication geometry are represented.
All working dimensions are visual millimetres; GLB exports use metres.
"""
import bpy, math, json, os
from mathutils import Vector, Quaternion
from pathlib import Path
OUT=Path(__file__).resolve().parent
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
for block in list(bpy.data.materials): bpy.data.materials.remove(block)
scene=bpy.context.scene
scene.unit_settings.system='METRIC'; scene.unit_settings.scale_length=.001

def status(msg, renders=None, complete=False):
    d={'message':msg,'model':'wane-concept.glb'}
    if renders: d['renders']=renders
    if complete: d['complete']=True
    (OUT/'status.json').write_text(json.dumps(d))

def linear(hexstr):
    h=hexstr.lstrip('#'); c=[int(h[i:i+2],16)/255 for i in (0,2,4)]
    return tuple(x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4 for x in c)+(1,)

def material(name,color,metal=0,rough=.4,emission=0):
    m=bpy.data.materials.new(name); m.diffuse_color=linear(color); m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF'); p.inputs['Base Color'].default_value=linear(color)
    p.inputs['Metallic'].default_value=metal; p.inputs['Roughness'].default_value=rough
    if emission:
        p.inputs['Emission Color'].default_value=linear(color); p.inputs['Emission Strength'].default_value=emission
    return m
def glass_material(name,color,rough=.16,alpha=.34):
    m=material(name,color,.05,rough)
    p=m.node_tree.nodes.get('Principled BSDF')
    if p.inputs.get('Transmission Weight'): p.inputs['Transmission Weight'].default_value=1.0
    elif p.inputs.get('Transmission'): p.inputs['Transmission'].default_value=1.0
    if p.inputs.get('Alpha'): p.inputs['Alpha'].default_value=alpha
    m.diffuse_color=linear(color)[:3]+(alpha,)
    m.surface_render_method='DITHERED'
    return m
shell=material('M_SHELL','#0E2B26',.62,.36)
cap=material('M_CAP','#303B36',.18,.32)
titanium=material('M_TITANIUM','#ADA99B',.83,.3)
black=material('M_RECESS','#111B18',.15,.48)
lens=material('M_LENS','#20332E',.27,.22)
window=material('M_WINDOW','#263A31',.25,.17)
pod_glass=glass_material('M_POD_GLASS','#B6D5C5',.12,.28)
flavour_liquid=glass_material('M_FLAVOUR_LIQUID','#A7D5BF',.08,.56)
nicotine_liquid=glass_material('M_NICOTINE_LIQUID','#C89455',.10,.58)
grip=material('M_REAR_GRIP','#183A31',.22,.62)
track=material('M_LIGHT_TRACK','#314A41',.35,.38)
light=material('M_LED_TEAL','#5FD3C4',.05,.26,2.4)
mark=material('M_ENGRAVING','#70857B',.48,.45)
# Tiny satin surface variation; only the explicit PBR values carry into GLB.
n=shell.node_tree.nodes; links=shell.node_tree.links
noise=n.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value=420; noise.inputs['Detail'].default_value=2
bump=n.new('ShaderNodeBump'); bump.inputs['Strength'].default_value=.1; bump.inputs['Distance'].default_value=.018
links.new(noise.outputs['Fac'],bump.inputs['Height']); links.new(bump.outputs['Normal'],n.get('Principled BSDF').inputs['Normal'])

root=bpy.data.objects.new('WANE_EXTERIOR_CONCEPT',None); scene.collection.objects.link(root)
root['description']='Nonfunctional exterior styling study. Internal packaging not validated.'
root['nominal_envelope_mm']='105 x 22 x 12'; root['revision']='01'

def empty(name,parent=root,loc=(0,0,0)):
    o=bpy.data.objects.new(name,None); scene.collection.objects.link(o); o.parent=parent; o.location=loc; o.empty_display_size=.8; return o
body=empty('WN-09_SHELL'); upper=empty('WN-04_UPPER_HOUSING'); pod=empty('WN-02_DUAL_POD'); mouth=empty('WN-01_MOUTHPIECE'); led=empty('WN-08_LIGHT_ASSEMBLY')

def finish(o,mat,parent):
    o.data.materials.append(mat); o.parent=parent
    for p in o.data.polygons:p.use_smooth=True
    return o

def profile(name,sections,mat,parent,exp=3.6,segments=96):
    # Each section = (height,width,depth); superellipse contours soften the sides.
    vs=[]; fs=[]
    for z,w,d in sections:
        for i in range(segments):
            a=2*math.pi*i/segments; c=math.cos(a); s=math.sin(a)
            vs.append((w*.5*math.copysign(abs(c)**(2/exp),c),d*.5*math.copysign(abs(s)**(2/exp),s),z))
    for k in range(len(sections)-1):
        for i in range(segments):
            j=(i+1)%segments; fs.append((k*segments+i,k*segments+j,(k+1)*segments+j,(k+1)*segments+i))
    fs.append(tuple(reversed(range(segments)))); fs.append(tuple((len(sections)-1)*segments+i for i in range(segments)))
    me=bpy.data.meshes.new(name); me.from_pydata(vs,[],fs); me.update()
    o=bpy.data.objects.new(name,me); scene.collection.objects.link(o); finish(o,mat,parent)
    o.data.polygons[-1].use_smooth=False; o.data.polygons[-2].use_smooth=False
    return o

def box(name,loc,size,mat,parent,r=.2):
    bpy.ops.mesh.primitive_cube_add(size=1,location=loc); o=bpy.context.object; o.name=name; o.dimensions=size
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    if r:
        mod=o.modifiers.new('Soft machined edges','BEVEL'); mod.width=r; mod.segments=5
        bpy.context.view_layer.objects.active=o; bpy.ops.object.modifier_apply(modifier=mod.name)
        mod=o.modifiers.new('Face normals','WEIGHTED_NORMAL'); mod.keep_sharp=True; mod.weight=50
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return finish(o,mat,parent)

def cylinder(name,loc,r,depth,mat,parent,rotation=(math.pi/2,0,0),vertices=96):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices,radius=r,depth=depth,location=loc,rotation=rotation)
    o=bpy.context.object; o.name=name; finish(o,mat,parent)
    b=o.modifiers.new('Edge highlight','BEVEL');b.width=.06;b.segments=3
    bpy.context.view_layer.objects.active=o;bpy.ops.object.modifier_apply(modifier=b.name)
    w=o.modifiers.new('Face normals','WEIGHTED_NORMAL');bpy.ops.object.modifier_apply(modifier=w.name)
    return o

def arc(name,center,r,width,start,end,mat,parent,segments=128):
    x,y,z=center; vs=[]; fs=[]
    for i in range(segments+1):
        a=start+(end-start)*i/segments
        for rr in (r-width/2,r+width/2):vs.append((x+rr*math.sin(a),y,z+rr*math.cos(a)))
    for i in range(segments):fs.append((2*i+2,2*i+3,2*i+1,2*i))
    me=bpy.data.meshes.new(name);me.from_pydata(vs,[],fs);me.update();o=bpy.data.objects.new(name,me);scene.collection.objects.link(o);return finish(o,mat,parent)

# Main housing: a continuous soft oval, with a small separate foot.
profile('SHELL_FRONT',[(3.15,20.6,10.6),(3.35,21.35,11.35),(3.9,21.9,11.9),(4.5,22,12),(80.6,22,12),(82.6,21.95,11.95),(83.3,21.7,11.7),(83.6,21.2,11.2)],shell,body)
profile('Base_shadow_seam',[(2.65,20.65,10.65),(3.28,20.65,10.65)],black,body)
profile('Base_heel',[(0,17.8,7.8),(.18,19.15,9.15),(.6,20.3,10.3),(1.25,21.2,11.2),(2.2,21.6,11.6),(2.75,21.35,11.35)],shell,body)
# The lower pod is deliberately transparent: two narrow cartridges make the
# two liquids legible without turning the front into a screen or a toy.
box('Pod_glass_outer',(0,-6.06,28.5),(17.4,.38,39.4),pod_glass,pod,.62)
box('Pod_frame_left',(-4.65,-6.23,28.5),(.55,.14,37.3),titanium,pod,.12)
box('Pod_frame_right',(4.65,-6.23,28.5),(.55,.14,37.3),titanium,pod,.12)
box('Pod_frame_spine',(0,-6.23,28.5),(.38,.14,37.3),titanium,pod,.08)
box('Pod_frame_top',(0,-6.23,47.0),(9.8,.14,.52),titanium,pod,.09)
box('Pod_frame_bottom',(0,-6.23,10.0),(9.8,.14,.52),titanium,pod,.09)
box('LIQUID_F_FLAVOUR',(-2.5,-6.31,28.2),(3.6,.16,31.8),flavour_liquid,pod,.34)
box('LIQUID_N', (2.5,-6.31,25.9),(3.6,.16,27.2),nicotine_liquid,pod,.34)
label_pod_left=box('Pod_flavour_marker',(-2.5,-6.42,43.0),(2.6,.02,.42),light,pod,.05)
box('Pod_nicotine_marker',(2.5,-6.42,40.6),(2.6,.02,.42),titanium,pod,.05)
pod['interface']='Two visible liquid chambers: flavour left, nicotine right.'
# A two-tone tactile field on the rear gives the body a quiet change in both
# colour and texture, inspired by a precision-machined device rather than a logo.
box('Rear_tactile_field',(0,6.005,59.5),(17.6,.09,39.0),grip,body,.55)
for z in range(43,78,3):
    box('Rear_micro_rib_%02d'%z,(0,6.09,z),(15.4,.018,.14),titanium,body,.03)
box('Rear_field_rule',(0,6.12,79.4),(15.5,.02,.25),titanium,body,.02)
profile('Upper_shadow_seam',[(83.55,20.65,10.65),(84.22,20.65,10.65)],black,upper)
profile('Satin_collar',[(84.0,21.1,11.1),(84.13,21.65,11.65),(84.55,21.65,11.65),(84.72,21.1,11.1)],titanium,upper)
profile('Upper_housing',[(84.7,20.9,10.9),(85,21.25,11.25),(87,21.1,11.1),(93,20.1,10.35),(96,19.35,9.75),(97.2,18.8,9.3)],shell,upper)
profile('Mouthpiece_shadow',[(97.15,18.1,8.65),(97.55,18.1,8.65)],black,mouth)
profile('Soft_mouthpiece',[(97.5,18.6,9.1),(98,18.6,9.1),(101.6,17.2,8.25),(103.5,15.6,7.45),(104.45,14.3,6.6),(104.85,13.25,5.6),(105,12,4.5)],cap,mouth,exp=3.1)
# Shallow dark inset on the top: cosmetic only, no connected airway.
box('Mouthpiece_top_inset',(0,0,105.02),(6.2,1.55,.06),black,mouth,.025)

# LED sits on the permanent front shell, below the removable upper seam.
# It is a small flush instrument dial, rather than a large glowing band.
cylinder('Dial_satin_rim',(0,-5.995,71.6),3.42,.25,titanium,led)
cylinder('Dial_dark_lens',(0,-6.14,71.6),3.15,.08,lens,led)
arc('Dial_unlit_track',(0,-6.196,71.6),2.65,.32,0,2*math.pi,track,led)
arc('STUDIO_LED_ARC',(0,-6.205,71.6),2.65,.30,math.radians(35),math.radians(325),light,led)
empty('LED_CENTER',led,(0,-6.23,71.6))
# No second decorative status dot: one visual signal only.
# Rear-only upper windows. These are visual inlays, not reservoirs.
for x,nm in [(-3.8,'Left'),(3.8,'Right')]:
    box(nm+'_window_bezel',(x,5.3,90.8),(2.65,.35,7.6),black,upper,.17)
    box(nm+'_window_lens',(x,5.505,90.8),(1.75,.07,6.4),window,upper,.034)
    box(nm+'_window_highlight',(x-.48,5.55,90.8),(.13,.01,5.3),mark,upper,.004)
empty('WINDOW',upper,(3.8,5.55,90.8))
# Discreet rear marking; keep the front free of branding.
font_path='/System/Library/Fonts/Supplemental/Arial.ttf'
font=bpy.data.fonts.load(font_path) if os.path.exists(font_path) else None

def label(name,text,loc,size,mat,parent,rotation=(math.pi/2,0,math.pi)):
    cu=bpy.data.curves.new(name,'FONT');cu.body=text;cu.size=size;cu.align_x='CENTER';cu.align_y='CENTER';cu.space_character=1.3
    if font:cu.font=font
    o=bpy.data.objects.new(name,cu);scene.collection.objects.link(o);o.location=loc;o.rotation_euler=rotation;o.parent=parent;cu.materials.append(mat);return o
label('Rear_signature','W A N E',(0,6.016,17),1.3,mark,body)
label('Rear_revision','O N E   /   0 1',(0,6.016,13.5),.55,mark,body)
# Underside cosmetic port. A visual cue, not a functional connector model.
box('Base_port_recess',(0,0,-.02),(7.4,2.5,.08),black,body,.035)
box('Base_port_tongue',(0,0,-.07),(4.9,.7,.035),titanium,body,.014)

# GLB uses +Y up and +Z front after Blender's default axis conversion.
def export(path,studio=True):
    bpy.ops.object.select_all(action='DESELECT')
    objects=[root]+list(root.children_recursive)
    # Convert the two type objects once; keeps the GLB self-contained.
    for o in objects:
        if o.type=='FONT':
            bpy.context.view_layer.objects.active=o;o.select_set(True);bpy.ops.object.convert(target='MESH');o.select_set(False)
    for o in [root]+list(root.children_recursive):
        if not studio and o.name.startswith('STUDIO_'):continue
        o.select_set(True)
    root.scale=(.001,.001,.001);bpy.context.view_layer.update()
    bpy.ops.export_scene.gltf(filepath=str(OUT/path),export_format='GLB',use_selection=True,export_yup=True,export_extras=True,export_materials='EXPORT')
    root.scale=(1,1,1);bpy.context.view_layer.update();bpy.ops.object.select_all(action='DESELECT')
export('wane-concept.glb')
export('wane-integration.glb',False)
status('02 / Exterior model ready. Preparing studio lighting and renders.')

# Studio staging in millimetre visual units.
floor_mat=material('Studio_chalk','#BDBDB0',0,.72)
bpy.ops.mesh.primitive_plane_add(size=2000,location=(0,0,-.15));floor=bpy.context.object;floor.name='STUDIO_FLOOR';floor.data.materials.append(floor_mat)
world=bpy.data.worlds.new('Soft studio') if not bpy.data.worlds else bpy.data.worlds[0];scene.world=world;world.use_nodes=True;world.node_tree.nodes['Background'].inputs[0].default_value=(.22,.27,.24,1);world.node_tree.nodes['Background'].inputs[1].default_value=.45

def area(name,loc,energy,size,color,target=(0,0,52),shape='DISK',size_y=None):
    data=bpy.data.lights.new(name,'AREA');data.energy=energy;data.shape=shape;data.size=size;data.color=color
    if size_y is not None:data.size_y=size_y
    o=bpy.data.objects.new(name,data);scene.collection.objects.link(o);o.location=loc;o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler();return o
# At this working scale, light energies are scaled to match distances.
area('Key_softbox',(-90,-110,170),420000,115,(1,.94,.84),shape='RECTANGLE',size_y=175)
area('Rim_strip',(85,45,145),510000,42,(.79,.91,1),shape='RECTANGLE',size_y=150)
area('Front_fill',(55,-130,60),125000,95,(.88,1,.95))
area('Top_silk',(-20,30,200),220000,90,(1,.98,.9))
cam_data=bpy.data.cameras.new('Studio_camera');cam=bpy.data.objects.new('Studio_camera',cam_data);scene.collection.objects.link(cam);scene.camera=cam
cam_data.type='ORTHO';cam_data.lens=75;cam_data.clip_end=5000

def camera(loc,target,scale):
    cam.location=loc;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=scale
camera((145,-245,150),(0,0,53),139)
scene.render.engine='CYCLES';scene.cycles.samples=40;scene.cycles.use_denoising=True
scene.cycles.max_bounces=6;scene.cycles.diffuse_bounces=3;scene.cycles.glossy_bounces=4
scene.render.resolution_x=1500;scene.render.resolution_y=1500;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
scene.view_settings.view_transform='AgX';scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=.3
# A restrained lens glow in renders only.
# Useful initial viewport when the blend is opened.
bpy.context.view_layer.objects.active=body
for scr in bpy.data.screens:
    for a in scr.areas:
        if a.type=='VIEW_3D':
            s=a.spaces.active;s.clip_end=5000;s.region_3d.view_distance=165;s.region_3d.view_location=(0,0,53);s.region_3d.view_rotation=cam.rotation_euler.to_quaternion();s.shading.type='MATERIAL';s.overlay.show_overlays=False
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'wane-studio.blend'))

def render(name):
    scene.render.filepath=str(OUT/name);bpy.ops.render.render(write_still=True)
render('01-pine-studio.png')
status('03 / Pine render ready. Rendering the three finishes.', ['01-pine-studio.png'])
# Create the finish family from the actual geometry, not duplicated images.
root.location.x=-31;root.rotation_euler.z=math.radians(-11)
originals=[root]+list(root.children_recursive)
def clone_finish(name,hexcol,x,angle):
    mapping={};newmat=shell.copy();newmat.name='M_SHELL_'+name;newmat.diffuse_color=linear(hexcol);newmat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=linear(hexcol)
    for old in originals:
        obj=old.copy()
        if old.data:obj.data=old.data.copy()
        scene.collection.objects.link(obj);mapping[old]=obj
    for old,obj in mapping.items():
        obj.parent=mapping.get(old.parent)
        if obj.type=='MESH':
            for slot in obj.material_slots:
                if slot.material==shell:slot.material=newmat
    nr=mapping[root];nr.name='FINISH_'+name;nr.location.x=x;nr.rotation_euler.z=math.radians(angle)
    return nr,list(mapping.values())
sage,sage_objs=clone_finish('SAGE','#8FA79B',0,0)
bone,bone_objs=clone_finish('BONE','#D9D4C7',31,11)
camera((120,-330,172),(0,0,51),154);scene.render.resolution_x=1800;scene.render.resolution_y=1400
render('02-finish-family.png')
status('04 / Finish family ready. Rendering the LED and rear details.', ['01-pine-studio.png','02-finish-family.png'])
for o in sage_objs+bone_objs:bpy.data.objects.remove(o,do_unlink=True)
root.location.x=0;root.rotation_euler.z=0
camera((65,-150,112),(0,-3,77),49);scene.render.resolution_x=1500;scene.render.resolution_y=1400
render('03-light-detail.png')
camera((-140,240,150),(0,0,53),136);scene.render.resolution_x=1500;scene.render.resolution_y=1500
render('04-rear-study.png')
camera((145,-245,150),(0,0,53),139)
scene.render.resolution_x=1500;scene.render.resolution_y=1500;scene.render.filepath=str(OUT/'01-pine-studio.png')
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'wane-studio.blend'))
metrics={'visual_envelope_mm':[105,22,12],'mesh_objects':sum(1 for o in root.children_recursive if o.type=='MESH'),'triangles':sum(sum(len(p.vertices)-2 for p in o.data.polygons) for o in root.children_recursive if o.type=='MESH'),'glb_bytes':(OUT/'wane-concept.glb').stat().st_size,'notes':'Exterior concept only; no internal feasibility validation. Static LED arc excluded from integration GLB.'}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
status('Ready for review / Drag to inspect the model. All four studio renders are below.', ['01-pine-studio.png','02-finish-family.png','03-light-detail.png','04-rear-study.png'],True)
print('WANE_BUILD_COMPLETE',json.dumps(metrics))
