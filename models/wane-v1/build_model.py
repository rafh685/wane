"""Wane exterior styling study. Run with Blender --background --python this_file.
No functional internal components or fabrication geometry are represented.
All working dimensions are visual millimetres; GLB exports use metres.
"""
import bpy, math, json, os, time, sys
from mathutils import Vector, Quaternion
from pathlib import Path
OUT=Path(__file__).resolve().parent
BUILD_VERSION=str(time.time_ns())
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
for block in list(bpy.data.materials): bpy.data.materials.remove(block)
scene=bpy.context.scene
scene.unit_settings.system='METRIC'; scene.unit_settings.scale_length=.001

def status(msg, renders=None, complete=False):
    d={'message':msg,'model':'wane-concept.glb','revision':BUILD_VERSION}
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
def glass_material(name,color,rough=.16,opacity=.3):
    m=material(name,color,0,rough)
    p=m.node_tree.nodes.get('Principled BSDF')
    # Alpha transparency stays legible with nested chambers in realtime GLB.
    p.inputs['Alpha'].default_value=opacity
    p.inputs['IOR'].default_value=1.46
    m.diffuse_color=linear(color)[:3]+(opacity,)
    m.surface_render_method='DITHERED'
    return m
shell=material('M_SHELL','#0E2B26',.55,.34)
cap=material('M_CAP','#15171A',0,.22)
titanium=material('M_TITANIUM','#A8ADAF',.8,.3)
black=material('M_RECESS','#121416',0,.42)
lens=material('M_LENS','#20332E',.27,.22)
pod_glass=glass_material('M_POD_SMOKE_PLASTIC','#626B75',.13,.22)
flavour_liquid=glass_material('M_FLAVOUR_LIQUID','#B5C4BD',.12,.18)
nicotine_liquid=glass_material('M_NICOTINE_LIQUID','#AD997E',.12,.22)
meniscus=glass_material('M_LIQUID_SURFACE','#AFB7B8',.1,.46)
partition=material('M_POD_PARTITION','#51565B',0,.3)
track=material('M_LIGHT_TRACK','#314A41',.35,.38)
light=material('M_LED_TEAL','#5FD3C4',.05,.26,2.4)
mark=material('M_ENGRAVING','#70857B',.48,.45)
wordmark=material('M_WORDMARK','#BBC8C0',.18,.48)

root=bpy.data.objects.new('WANE_EXTERIOR_CONCEPT',None); scene.collection.objects.link(root)
root['description']='Nonfunctional exterior styling study. Internal packaging not validated.'
root['nominal_envelope_mm']='105 x 22 x 12'; root['revision']='03-top-pod'

def empty(name,parent=root,loc=(0,0,0)):
    o=bpy.data.objects.new(name,None); scene.collection.objects.link(o); o.parent=parent; o.location=loc; o.empty_display_size=.8; return o
body=empty('WN-09_SHELL')
removable=empty('POD_REMOVABLE_ASSEMBLY')
pod=empty('WN-02_DUAL_POD',removable)
mouth=empty('WN-01_MOUTHPIECE',removable)
led=empty('WN-08_LIGHT_ASSEMBLY')

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
profile('SHELL_FRONT',[(3.15,20.6,10.6),(3.35,21.35,11.35),(3.9,21.9,11.9),(4.5,22,12),(81.8,22,12),(82.6,21.95,11.95),(83.1,21.6,11.6),(83.35,21.15,11.15)],shell,body)
profile('Base_shadow_seam',[(2.65,20.65,10.65),(3.28,20.65,10.65)],black,body)
profile('Base_heel',[(0,17.8,7.8),(.18,19.15,9.15),(.6,20.3,10.3),(1.25,21.2,11.2),(2.2,21.6,11.6),(2.75,21.35,11.35)],shell,body)
# Exterior-only cartridge study. Both chambers and the mouthpiece lift together.
# The seams describe the removable silhouette, without engineering the mechanism.
profile('Body_top_recess',[(83.34,20.6,10.6),(83.53,20.6,10.6)],black,body)
profile('Pod_lower_rim',[(83.62,20.3,10.3),(84.4,20.3,10.3)],black,pod)
pod_shell=profile('Smoked_plastic_pod',[(84.32,20.0,10.0),(84.7,20.65,10.65),(85.25,20.8,10.8),(92.5,20.3,10.4),(93.6,19.95,10.0),(94.05,19.5,9.55)],pod_glass,pod)
# Inner surface gives the clear visual shell a plastic-wall highlight.
solid=pod_shell.modifiers.new('Visible plastic wall','SOLIDIFY');solid.thickness=.30;solid.offset=-1
bpy.context.view_layer.objects.active=pod_shell;bpy.ops.object.modifier_apply(modifier=solid.name)
box('CHAMBER_DIVIDER',(2.05,0,89.0),(.38,8.3,8.6),partition,pod,.16)
box('LIQUID_F_FLAVOUR',(-3.38,0,88.48),(9.95,7.75,6.5),flavour_liquid,pod,.7)
box('LIQUID_N',(5.58,0,87.95),(6.02,7.75,5.44),nicotine_liquid,pod,.7)
box('Flavour_surface',(-3.38,0,91.70),(9.65,7.45,.035),meniscus,pod,.015)
box('Nicotine_surface',(5.58,0,90.64),(5.72,7.45,.035),meniscus,pod,.015)
empty('WINDOW',pod,(0,-5.43,89.0))
pod['interface']='Top cartridge with two visual chambers, under one smoky plastic housing.'
removable['animation']='Lift the whole assembly vertically; mouthpiece stays attached to pod.'
profile('Mouthpiece_shadow',[(93.98,19.45,9.50),(94.33,19.45,9.50)],black,mouth)
profile('Soft_mouthpiece',[(94.25,19.95,10.0),(94.65,20.3,10.3),(95.5,20.4,10.4),(99.4,19.4,9.6),(102.2,17.8,8.4),(103.65,16.4,7.35),(104.5,15.0,6.05),(104.9,13.5,4.9),(105,12.1,4.0)],cap,mouth,exp=3.1)
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
# Discreet rear marking; keep the front free of branding.
font_path='/System/Library/Fonts/Supplemental/Arial.ttf'
font=bpy.data.fonts.load(font_path) if os.path.exists(font_path) else None

def label(name,text,loc,size,mat,parent,rotation=(math.pi/2,0,math.pi)):
    cu=bpy.data.curves.new(name,'FONT');cu.body=text;cu.size=size;cu.align_x='CENTER';cu.align_y='CENTER';cu.space_character=1.3
    if font:cu.font=font
    o=bpy.data.objects.new(name,cu);scene.collection.objects.link(o);o.location=loc;o.rotation_euler=rotation;o.parent=parent;cu.materials.append(mat);return o
label('Rear_signature','W A N E',(0,6.04,18),2.2,wordmark,body)
label('Rear_revision','O N E   /   0 3',(0,6.016,13.5),.55,mark,body)
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
scene.cycles.transmission_bounces=8;scene.cycles.max_bounces=12
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
if '--branding' in sys.argv:
    camera((-80,245,117),(0,0,52),130)
    render('05-wordmark-rear.png')
    status('Ready / Larger, clearer WANE lettering. Rotation speed increased by 10%.', ['01-pine-studio.png','02-finish-family.png','03-light-detail.png','04-pod-removed.png','05-wordmark-rear.png'],True)
    sys.exit(0)
render('01-pine-studio.png')
if '--draft' in sys.argv:
    status('Updated model / Smooth shell and removable smoky plastic top pod.', ['01-pine-studio.png'],True)
    sys.exit(0)
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
status('04 / Finish family ready. Rendering the pod close-up and removal view.', ['01-pine-studio.png','02-finish-family.png'])
for o in sage_objs+bone_objs:bpy.data.objects.remove(o,do_unlink=True)
root.location.x=0;root.rotation_euler.z=0
camera((48,-155,110),(0,-1,90),36);scene.render.resolution_x=1500;scene.render.resolution_y=1400
render('03-light-detail.png')
removable.location.z=16
camera((70,-245,139),(0,0,61),148);scene.render.resolution_x=1500;scene.render.resolution_y=1500
render('04-pod-removed.png')
removable.location.z=0
camera((145,-245,150),(0,0,53),139)
scene.render.resolution_x=1500;scene.render.resolution_y=1500;scene.render.filepath=str(OUT/'01-pine-studio.png')
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'wane-studio.blend'))
metrics={'visual_envelope_mm':[105,22,12],'mesh_objects':sum(1 for o in root.children_recursive if o.type=='MESH'),'triangles':sum(sum(len(p.vertices)-2 for p in o.data.polygons) for o in root.children_recursive if o.type=='MESH'),'glb_bytes':(OUT/'wane-concept.glb').stat().st_size,'notes':'Exterior concept only; no internal feasibility validation. Static LED arc excluded from integration GLB.'}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
status('Ready / Smooth body, removable smoky plastic top pod and two visible chambers.', ['01-pine-studio.png','02-finish-family.png','03-light-detail.png','04-pod-removed.png'],True)
print('WANE_BUILD_COMPLETE',json.dumps(metrics))
