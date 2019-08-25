// Keyboard control
#include <GLFW\glfw3.h>

#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <assert.h>
#include <time.h>

#include <optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

// Nvidia function
#include "sutil\sutil.h"
#include <optix.h>
#include <Camera.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw_gl2.h>

// My class
#include "model.h"
#include "helper_function.h"
#include "optixPathTracer.h"
#include <random.h>


using namespace std;
using namespace optix;

#pragma warning( disable : 4996 )

//// Model path
string full_path = "./model/sponza/sponza.obj";
string mtl_path  = "./model/sponza/sponza.mtl";


// Model path
//string full_path = "./model/cornell_box/cornellbox.obj";
//string mtl_path = "./model/cornell_box/cornellbox.mtl";
//// 
//string full_path = "./model/church/church.obj";
//string mtl_path  = "./model/church/church.mtl";
//
//string full_path = "./model/bunny/bunny.obj";
//string mtl_path  = "./model/bunny/bunny.mtl";
//
//string full_path = "./model/bedroom/bedroom.obj";
//string mtl_path = "./model/bedroom/bedroom.mtl";
////
//string full_path = "./model/restroom/restroom.obj";
//string mtl_path = "./model/restroom/restroom.mtl";
//
//string full_path = "./model/vokselia_spawn/vokselia_spawn2.obj";
//string mtl_path = "./model/vokselia_spawn/vokselia_spawn2.mtl";
//
//string full_path = "./model/bmw/bmw2.obj";
//string mtl_path = "./model/bmw/bmw2.mtl";
//
//string full_path = "./model/breakfast_room/breakfast_room.obj";
//string mtl_path = "./model/breakfast_room/breakfast_room.mtl";
//
//string full_path = "./model/conference/conference.obj";
//string mtl_path = "./model/conference/conference.mtl";


//// gallery
//string full_path = "./model/gallery/gallery2.obj";
//string mtl_path = "./model/gallery/gallery2.mtl";

// san-miguel
//string full_path = "./model/San_Miguel/san-miguel2.obj";
//string mtl_path = "./model/San_Miguel/san-miguel2.mtl";

//// hair-ball
//string full_path = "./model/hairball/hairball2.obj";
//string mtl_path = "./model/hairball/hairball2.mtl";

//// fireplace_room
//string full_path = "./model/fireplace_room/fireplace_room.obj";
//string mtl_path = "./model/fireplace_room/fireplace_room.mtl";


const std::string outputImage1 = "C:/Users/NTNU_CGLab/Desktop/demo/1sppImage.ppm";
const std::string outputImageNormal = "C:/Users/NTNU_CGLab/Desktop/demo/1sppImage_Normal.ppm";
const std::string outputImageDepth = "C:/Users/NTNU_CGLab/Desktop/demo/1sppImage_Depth.ppm";

const std::string outputImage2 = "C:/Users/NTNU_CGLab/Desktop/demo/400sppImage.ppm";


const char* SAMPLE_NAME = "optixPathTracer";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context context = 0;

//window
int WIDTH =  1920,
HEIGHT = 1080;

// phong 
float KA = 0.1;

int SAMPLE_PIXEL	= 1;
int RR_BEGIN_DEPTH	= 1;
int TRACE_DEPTH = INFINITY;
int FRAME_NUMBER	= 1; 
bool LERP_BOL		= false;
bool window_128		= true;


//light
float3 LIGHT_POS   = make_float3( 1.5f,  0.1f, -0.5f);
//float3 LIGHT_POS = make_float3(1.615f, 0.115f, -0.889f);

// sponza
//float3 LIGHT_POS = make_float3(1.6f, 0.5f/*10.02.0f*/, -0.6f);
//float3 LIGHT_POS = make_float3(-5.0f, 10.0f, -6.0f);

//cornell box
//float3 LIGHT_POS = make_float3(-0.0f, 1.8f, -0.0f);

// forest
//float3 LIGHT_POS = make_float3(-1.0f, 5.0f/*2.0f*/, 0.0f);

// bmw
//float3 LIGHT_POS = make_float3(0.0, 3.8f, -0.4f);

//conference
//float3 LIGHT_POS = make_float3(0.7, 0.4f, 0.0f);


//float3 LIGHT_POS = make_float3(-0.0f, 0.2f, 0.0f);
//float3 LIGHT_POS   = make_float3(0.0f, 10.0, -10.0f);


//sponza
//float3 LIGHT_COLOR = make_float3(10.0f, 10.0f, 10.0f);
float3 LIGHT_COLOR = make_float3(1.0f, 1.0f, 1.0f);
//float3 LIGHT_COLOR = make_float3(5.0f, 5.0f, 5.0f);

//float3 LIGHT_COLOR = make_float3(0.0f);


//camera
float3 CAMERA_POS    = make_float3(0.0f, 0.0f, 0.6f);
float3 CAMERA_LOOKAT = make_float3(0.0f, 0.0f, 0.5f);
float3 CAMERA_UP     = make_float3(0.0f, 1.0f, 0.0f);

// sponza bounding box
float3 bounding_max = make_float3(7.030891, 5.583724, 4.318072);
float3 bounding_min = make_float3(-7.503695, -0.493917, -4.620340);


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------


void glfwRun(GLFWwindow* window, sutil::Camera& camera, Buffer outputbuffer);



void windowSizeCallback(GLFWwindow* window, int w, int h);



void setMaterial(
	GeometryInstance& gi,
	Material material,
	const std::string& color_name,
	const float3& color)
{
	gi->addMaterial(material);
	gi[color_name]->setFloat(color);
}

GeometryInstance createParallelogram(
	Program pgram_intersection,
	Program pgram_bounding_box,
	const float3& anchor,
	const float3& offset1,
	const float3& offset2)
{
	Geometry parallelogram = context->createGeometry();
	parallelogram->setPrimitiveCount(1u);
	parallelogram->setIntersectionProgram(pgram_intersection);
	parallelogram->setBoundingBoxProgram(pgram_bounding_box);

	float3 normal = normalize(cross(offset1, offset2));
	float d = dot(normal, anchor);
	float4 plane = make_float4(normal, d);

	float3 v1 = offset1 / dot(offset1, offset1);
	float3 v2 = offset2 / dot(offset2, offset2);

	parallelogram["plane"]->setFloat(plane);
	parallelogram["anchor"]->setFloat(anchor);
	parallelogram["v1"]->setFloat(v1);
	parallelogram["v2"]->setFloat(v2);

	GeometryInstance gi = context->createGeometryInstance();
	gi->setGeometry(parallelogram);
	return gi;
}



inline Buffer getOutputBuffer()
{
	return context["output_buffer"]->getBuffer();
}
Buffer getNormalBuffer()
{
	return context["input_normal_buffer"]->getBuffer();
}

inline Buffer getdepthBuffer()
{
	return context["input_depth_buffer"]->getBuffer();
}

void destroyContext()
{
	if (context)
	{
		context->destroy();
		context = 0;
	}
}

void convertNormalsToColors(
	Buffer& normalBuffer)
{
	float* data = reinterpret_cast<float*>(normalBuffer->map());

	RTsize width, height;
	normalBuffer->getSize(width, height);

	RTsize size = width * height;
	for (size_t i = 0; i < size; ++i)
	{
		const float r = *(data + 3 * i);
		const float g = *(data + 3 * i + 1);
		const float b = *(data + 3 * i + 2);

		*(data + 3 * i) = std::abs(r);
		*(data + 3 * i + 1) = std::abs(g);
		*(data + 3 * i + 2) = std::abs(b);
	}

	normalBuffer->unmap();
}



//------------------------------------------------------------------------------
//
//  Context functions
//
//------------------------------------------------------------------------------

void createContext()
{

	// Initial context
	context = Context::create();

	// Set up context
	context->setEntryPointCount(1);
	context->setRayTypeCount(2);		// [0]= eye, [1] = shadow, [3]=transparency

	// Note: usually path tracing sample does not need a big stack size even with high ray depths,
	// beacuse rays are not shot recursively,
	// but this sample has alph texture,need a big stack size.
	context->setStackSize(3000);

	context[ "pathtrace_ray_type"	     ]->setUint(0u);
	context[ "pathtrace_shadow_ray_type" ]->setUint(1u);

	context[ "rr_begin_depth" ]->setUint(RR_BEGIN_DEPTH);
	
	


	context[ "scene_epsilon"  ]->setFloat(1.e-3f);
	context[ "frame_number"   ]->setUint(FRAME_NUMBER);
	context[ "lerp_bol"       ]->setInt(LERP_BOL);

	context["Ka"]->setFloat(make_float3(KA));

	// Buffer
	Buffer buffer = context->createBuffer(
		RT_BUFFER_OUTPUT, 
		RT_FORMAT_FLOAT4, 
		WIDTH, 
		HEIGHT
	);
	context[ "output_buffer" ]->set(buffer);


	// Buffer
	Buffer normal_buffer = context->createBuffer(
		RT_BUFFER_INPUT_OUTPUT,
		RT_FORMAT_FLOAT4,
		WIDTH,
		HEIGHT
	);
	context["input_normal_buffer"]->set(normal_buffer);

	// Buffer
	Buffer depth_buffer = context->createBuffer(
		RT_BUFFER_INPUT_OUTPUT,
		RT_FORMAT_FLOAT4,
		WIDTH,
		HEIGHT
	);
	context["input_depth_buffer"]->set(depth_buffer);


//	// Random Buffer
//	Buffer rnd_buffer = context->createBuffer(
//		RT_BUFFER_INPUT_OUTPUT,
//		RT_FORMAT_INT,
//		WIDTH,
//		HEIGHT
//	);
//	context["rnd_buffer"]->setBuffer(rnd_buffer);
//
//	int spp = 0;
//	
//	uint* seeds = reinterpret_cast<uint*>(rnd_buffer->map());
//	for (unsigned int i = 0; i < WIDTH*HEIGHT; ++i) {
//		seeds[i] = random1u() % 2;
//		//printf("%d\n", seeds[i]);
//
//		//seeds[i].y= random1u() % 2;
//		spp = spp + seeds[i];// .x + seeds[i].y;
//		if (spp > 1024) {
//			seeds[i] = 0;
//		}
//		//printf("%d\n", seeds[i]);
////		printf("%d, %d\n", seeds[i].x,seeds[i].y);
//	}
//	rnd_buffer->unmap();

	// Ray generate program
	{	
		std::string ptx_path(ptxPath("optixPathTracer.cu"));
		//std::string ptx_path(ptxPath("optixPathTracer2.cu"));
		context->setRayGenerationProgram(0,	context->createProgramFromPTXFile(ptx_path, "pathtrace_camera"));
		context->setExceptionProgram(0, context->createProgramFromPTXFile(ptx_path, "exception"));
		
		// Miss program
		//context->setMissProgram(0, context->createProgramFromPTXFile(ptx_path, "miss"));

		
		context->setMissProgram(0, context->createProgramFromPTXFile(ptx_path, "envmap_miss"));

		const std::string texture_filename = "";// "./model/Daylight Box.ppm";
		context["envmap"]->setTextureSampler(sutil::loadTexture(context, texture_filename, optix::make_float3(0.0f, 0.0f, 0.0f)));


		context["trace_depth"		]->setUint(TRACE_DEPTH);
		context[ "sqrt_num_samples" ]->setUint(SAMPLE_PIXEL);
		context[ "bad_color"        ]->setFloat(1000000.0f, 0.0f, 1000000.0f); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
		context[ "bg_color"         ]->setFloat(make_float3(0.0f,0.0f,0.0f));

	}

}

void createGeometry()
{
	// Light buffer
	ParallelogramLight light;
	light.corner = LIGHT_POS;
	//printf("%f,%f,%f\n", LIGHT_POS.x, LIGHT_POS.y, LIGHT_POS.z);
	//printf("%f,%f,%f\n", LIGHT_COLOR.x, LIGHT_COLOR.y, LIGHT_COLOR.z);



	//// sponza light
	light.v1 = make_float3(20.0f, 0.0f, 0.0f);
	light.v2 = make_float3(0.0f , 0.0f, 20.0f);

	//// bmw
	//light.v1 = make_float3(-2.0f, 0.0f, 0.0f);
	//light.v2 = make_float3(0.0f, 0.0f, 2.0f);
	light.normal = normalize(-cross(light.v1, light.v2));
	light.emission = LIGHT_COLOR;
	//make_float3(15.0f, 15.0f, 5.0f);

	Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(ParallelogramLight));
	light_buffer->setSize(1u);
	memcpy(light_buffer->map(), &light, sizeof(light));
	light_buffer->unmap();
	context["lights"]->setBuffer(light_buffer);

	

	// We use the base Mesh class rather than OptiXMesh, so we can customize materials below
	// for different passes.
	Mesh mesh;
	MeshLoader loader(full_path);// load file. It could be .obj or .ply
	loader.scanMesh(mesh);



	std::map<std::string, int> material_map;
	std::vector<material_t> material_textures;
	std::ifstream matIStream(mtl_path);

	LoadMtl(material_map, material_textures, matIStream);
	std::vector<material_t> vec_texture = setuptexture(material_map, material_textures);

	MeshBuffers buffers;
	setupMeshLoaderInputs(context, buffers, mesh);

	loader.loadMesh(mesh);


	// Translate to OptiX geometry
	const std::string path = ptxPath("triangle_mesh.cu");
	Program bounds_program = context->createProgramFromPTXFile(path, "mesh_bounds");
	Program intersection_program = context->createProgramFromPTXFile(path, "mesh_intersect");
	//printf("%f, %f, %f\n", mesh.bbox_max[0], mesh.bbox_max[1], mesh.bbox_max[2]);
	//printf("%f, %f, %f\n", mesh.bbox_min[0], mesh.bbox_min[1], mesh.bbox_min[2]);

	optix::Geometry geometry = context->createGeometry();
	geometry[ "vertex_buffer"   ]->setBuffer(buffers.positions);
	geometry[ "normal_buffer"   ]->setBuffer(buffers.normals);
	geometry[ "texcoord_buffer" ]->setBuffer(buffers.texcoords);
	geometry[ "material_buffer" ]->setBuffer(buffers.mat_indices);
	geometry[ "index_buffer"    ]->setBuffer(buffers.tri_indices);
	geometry->setPrimitiveCount(mesh.num_triangles);
	geometry->setBoundingBoxProgram(bounds_program);
	geometry->setIntersectionProgram(intersection_program);

	// Materials have different hit programs depending on pass.
	std::string ptx = ptxPath("optixPathTracer.cu");
	//std::string ptx = ptxPath("optixPathTracer2.cu");
	Program diffuse_ch = context->createProgramFromPTXFile(ptx, "diffuse");
	Program diffuse_ah = context->createProgramFromPTXFile(ptx, "shadow");
	Program texture_ch = context->createProgramFromPTXFile(ptx, "texture_hit");
	Program alpha_texture_ch = context->createProgramFromPTXFile(ptx, "alpha_texture_hit");
	Program mirror_texture_ch = context->createProgramFromPTXFile(ptx, "mirror");

	std::vector< optix::Material > optix_materials;
	for (int i = 0; i < mesh.num_materials; ++i) {

		optix::Material material = context->createMaterial();
		material->setClosestHitProgram(0, diffuse_ch);
		material->setAnyHitProgram(1, diffuse_ah);

		std::string spec_tex = vec_texture[i].specular_texname;
		std::string diff_tex = vec_texture[i].diffuse_texname;
		std::string alph_tex = vec_texture[i].alpha_texname;
		std::string bump_tex = vec_texture[i].bump_texname;
		std::string emss_tex = vec_texture[i].emssion_texture;

		material["diffuse_color"]->set3fv(mesh.mat_params[i].Kd);
		material["specular_color"]->set3fv(mesh.mat_params[i].Ks);
		//context["Ka"]->set3fv(mesh.mat_params[i].Ka);
	
		material[ "d_map"    ]->setTextureSampler(sutil::loadTexture(context, alph_tex, make_float3(1.0f)));
		material[ "Kd_map"   ]->setTextureSampler(sutil::loadTexture(context, diff_tex, make_float3(1.0f)));
		material[ "Ks_map"   ]->setTextureSampler(sutil::loadTexture(context, spec_tex, make_float3(1.0f)));
		material[ "bump_map" ]->setTextureSampler(sutil::loadTexture(context, bump_tex, make_float3(1.0f)));
		//material[ "Ke_map"   ]->setTextureSampler(sutil::loadTexture(context, emss_tex, make_float3(0.0f)));

		if (diff_tex != "") {material->setClosestHitProgram(0, texture_ch      );}
		if (alph_tex != "") {material->setClosestHitProgram(0, alpha_texture_ch);}
		


		if(mesh.mat_params[i].name=="grey_and_white_room:Mirror"){ material->setClosestHitProgram(0, mirror_texture_ch); }
		optix_materials.push_back(material);
	}


	optix::GeometryInstance geom_instance = context->createGeometryInstance(
		geometry,
		optix_materials.begin(),
		optix_materials.end()
	);
	unmap(buffers, mesh);
	

	GeometryGroup shadow_group = context->createGeometryGroup();
	shadow_group->addChild(geom_instance);
	shadow_group->setAcceleration(context->createAcceleration("TriangleKdTree"));

	
	context["top_shadower"]->set(shadow_group);
	//context["top_object"]->set(shadow_group);



	//bounding_max = make_float3(mesh.bbox_max[0], mesh.bbox_max[1], mesh.bbox_max[2]);
	//bounding_min= make_float3(mesh.bbox_min[0], mesh.bbox_min[1], mesh.bbox_min[2]);

	//// Light
	//// Set up parallelogram programs
	//std::string ptx2 = ptxPath("parallelogram.cu");
	//Program pgram_bounding_box = context->createProgramFromPTXFile(ptx2, "bounds");
	//Program pgram_intersection = context->createProgramFromPTXFile(ptx2, "intersect");

	//// create geometry instances
	//std::vector<GeometryInstance> gis;


	//gis.push_back(createParallelogram(
	//	pgram_intersection,
	//	pgram_bounding_box,
	//	LIGHT_POS,
	//	light.v1,
	//	light.v2));


	//Material diffuse_light = context->createMaterial();
	//Program diffuse_em = context->createProgramFromPTXFile(ptx, "diffuseEmitter");
	//diffuse_light->setClosestHitProgram(0, diffuse_em);

	//setMaterial(gis.back(), diffuse_light, "emission_color", make_float3(1.f));

	//shadow_group->addChild(gis[0]);

	//// Create geometry group
	//GeometryGroup geometry_group = context->createGeometryGroup(gis.begin(), gis.end());
	//geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
	context["top_object"]->set(shadow_group);


}


// ------------------------------------------------------------------------------
//
//  GLFW callbacks
//
//------------------------------------------------------------------------------

struct CallbackData
{
	sutil::Camera& camera;
	int& accumulation_frame;
};

void  keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	bool handled = false;
	bool light_changed = false;
	CallbackData* cb = static_cast<CallbackData*>(glfwGetWindowUserPointer(window));

	if (action == GLFW_PRESS)
	{
		switch (key)
		{
		case GLFW_KEY_Q:
		case GLFW_KEY_ESCAPE:
			if (context)
				context->destroy();
			if (window)
				glfwDestroyWindow(window);
			glfwTerminate();
			exit(EXIT_SUCCESS);

		case(GLFW_KEY_S):
		{
			
			std::cerr << "Saving current frame to '" << outputImage1 << "'\n";
			displayBufferPPM(outputImage1.c_str(), getOutputBuffer()->get(), false);
			std::cerr << "Saving current frame to '" << outputImageNormal << "'\n";
			displayBufferPPM(outputImageNormal.c_str(), getNormalBuffer()->get(), false);
			std::cerr << "Saving current frame to '" << outputImageDepth << "'\n";
			displayBufferPPM(outputImageDepth.c_str(), getdepthBuffer()->get(), false);
			handled = true;
			break;
		}
		case(GLFW_KEY_X):
		{

			std::cerr << "Saving current frame to '" << outputImage2 << "'\n";
			displayBufferPPM(outputImage2.c_str(), getOutputBuffer()->get(), false);
			handled = true;
			break;
		}
		case(GLFW_KEY_F):
		{
			//CallbackData* cb = static_cast<CallbackData*>(glfwGetWindowUserPointer(window));
			//CAMERA_POS = make_float3(0.0f, 0.0f, 0.6f);
			//CAMERA_LOOKAT = make_float3(0.0f, 0.0f, 0.5f);
			//CAMERA_UP = make_float3(0.0f, 1.0f, 0.0f);
			//cb->camera.reset_lookat();
			//cb->accumulation_frame = 0;
			WIDTH = 128;
			HEIGHT = 128;
			glfwSetWindowSize(window, (int)WIDTH, (int)HEIGHT);
			glfwSetWindowSizeCallback(window, windowSizeCallback);
			std::cerr << "reset window\n";
			//LIGHT_POS = make_float3(1.5f, 0.1f, -0.3f);
			FRAME_NUMBER = 0;
			handled = true;
			break;
		}
		case(GLFW_KEY_7): 
		{
			LIGHT_POS.x=LIGHT_POS.x + 0.10;
			light_changed = true;
			break;
		}
		case(GLFW_KEY_1):
		{
			LIGHT_POS.x=LIGHT_POS.x - 0.10;
			light_changed = true;
			break;
		}
		case(GLFW_KEY_8):
		{
			LIGHT_POS.y=LIGHT_POS.y + 0.10;
			light_changed = true;
			break;
		}
		case(GLFW_KEY_2):
		{
			LIGHT_POS.y =LIGHT_POS.y - 0.10;
			light_changed = true;
			break;
		}
		case(GLFW_KEY_9):
		{
			LIGHT_POS.z = LIGHT_POS.z + 0.10;
			light_changed = true;
			break;
		}
		case(GLFW_KEY_3):
		{
			LIGHT_POS.z = LIGHT_POS.z - 0.10;
			light_changed = true;
			break;
		}
		case(GLFW_KEY_L):
		{
			LIGHT_COLOR.x = LIGHT_COLOR.x + 1;
			LIGHT_COLOR = make_float3(LIGHT_COLOR.x);
			light_changed = true;
			break;
		}
		case(GLFW_KEY_D):
		{
			LIGHT_COLOR.x = LIGHT_COLOR.x - 1;
			LIGHT_COLOR = make_float3(LIGHT_COLOR.x);
			light_changed = true;
			break;
		}

		case(GLFW_KEY_N):
		{
			glfwRun(window, cb->camera, getNormalBuffer());
			break;
		}
		case(GLFW_KEY_O):
		{
			glfwRun(window, cb->camera, getOutputBuffer());
			break;
		}
		case(GLFW_KEY_G):
		{
			glfwRun(window, cb->camera , getdepthBuffer());
			break;
		}
		}
	}
		//light change
		if (light_changed) {
			ParallelogramLight light;
			light.corner = LIGHT_POS;
			//printf("%f,%f,%f\n", LIGHT_POS.x, LIGHT_POS.y, LIGHT_POS.z);
			//// plane light
			//light.v1 = make_float3(1.0f, 0.0f, 0.0f);
			//light.v2 = make_float3(0.0f, 0.0f, 1.0f);

			// point light
			light.v1 = make_float3(5.0f, 0.0f, 0.0f);
			light.v2 = make_float3(0.0f, 0.0f, 5.0f);
			light.normal = normalize(-cross(light.v1, light.v2));
			light.emission = LIGHT_COLOR;
				//make_float3(15.0f, 15.0f, 5.0f);

			Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
			light_buffer->setFormat(RT_FORMAT_USER);
			light_buffer->setElementSize(sizeof(ParallelogramLight));
			light_buffer->setSize(1u);
			memcpy(light_buffer->map(), &light, sizeof(light));
			light_buffer->unmap();
			context["lights"]->setBuffer(light_buffer);

			FRAME_NUMBER = 0;
		}
	if (!handled) {
		// forward key event to imgui
		ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
	}
}

void windowSizeCallback(GLFWwindow* window, int w, int h)
{
	if (w < 0 || h < 0) return;

	const unsigned width = (unsigned)w;
	const unsigned height = (unsigned)h;

	CallbackData* cb = static_cast<CallbackData*>(glfwGetWindowUserPointer(window));
	if (cb->camera.resize(width, height)) {
		cb->accumulation_frame = 0;
	}

	WIDTH = width;
	HEIGHT = height;
	sutil::resizeBuffer(getOutputBuffer(), width, height);
	glfwSwapInterval(0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);
	glViewport(0, 0, width, height);
}


//------------------------------------------------------------------------------
//
// GLFW setup and run 
//
//------------------------------------------------------------------------------

GLFWwindow* glfwInitialize()
{
	GLFWwindow* window = sutil::initGLFW();

	// Note: this overrides imgui key callback with our own.  We'll chain this.
	glfwSetKeyCallback(window, keyCallback);

	glfwSetWindowSize(window, (int)WIDTH, (int)HEIGHT);
	glfwSetWindowSizeCallback(window, windowSizeCallback);

	return window;
}

int rotate2 = 0;
//float y = 1.0;
void setCameraPostition(sutil::Camera& camera)
{
	float w_d = WIDTH / 128;
	float h_d = HEIGHT / 128;
	//rotate camera
	camera.rotate(5.0*w_d, 0.0);
	rotate2++;
	if (rotate2 == 20)
	{
		rotate2 = 0;
		//screen mov x,screen mov y,left_button_down,  right_button_down,  middle_button_down 
		camera.process_mouse(1.0, 1.0, 0, 0, 1);//ON
												//y = y + 500;
		camera.process_mouse(1.0, 1000 * h_d, 0, 0, 1);//OFF

	}

}

void setNormalMatrix(sutil::Camera& camera) {

	float3 camera_u = context["U"]->getFloat3();
	float3 camera_v = context["V"]->getFloat3();
	float3 camera_w = context["W"]->getFloat3();
	float3 camera_lookat = camera.getloogkat();



	const Matrix4x4 current_frame_inv = Matrix4x4::fromBasis(
		normalize(camera_u),
		normalize(camera_v),
		normalize(-camera_w),
		camera_lookat).inverse();

	Matrix3x3 normal_matrix = make_matrix3x3(current_frame_inv);

	context["normal_matrix"]->setMatrix3x3fv(false, normal_matrix.getData());

}


void glfwRun(GLFWwindow* window, sutil::Camera& camera, Buffer outputbuffer)
{

	// Initialize GL state
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glViewport(0, 0, WIDTH, HEIGHT);

	unsigned int frame_count = 0;


	//unsigned int accumulation_frame = 0;

	// Expose user data for access in GLFW callback functions when the window is resized, etc.
	// This avoids having to make it global.
	CallbackData cb = { camera,   FRAME_NUMBER };
	glfwSetWindowUserPointer(window, &cb);

	int image_number = 0;
	int noise_number = 0;// 2580;// 2692;// 2505;// FHD
	//					// 2522;// HD

	// sponza bounding box
	float3 max = bounding_max;// make_float3(7.030891, 5.583724, 4.318072);
	float3 min = bounding_min;// make_float3(-7.503695, -0.493917, -4.620340);

	//// church bounding box
	//float3 max = make_float3(0.531050, 1.964690, 1.258819);
	//float3 min = make_float3(-0.531050, -0.051421, -1.258819);

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		ImGui_ImplGlfwGL2_NewFrame();

		ImGuiIO& io = ImGui::GetIO();

		// Let imgui process the mouse first
		if (!io.WantCaptureMouse) {

			double x, y, z;
			glfwGetCursorPos(window, &x, &y);


			if (camera.process_mouse((float)x, (float)y, ImGui::IsMouseDown(0), ImGui::IsMouseDown(1), ImGui::IsMouseDown(2))) {
				FRAME_NUMBER = 0;

				setNormalMatrix(camera);
			}
		}

		// imgui pushes
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
		ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.6f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 2.0f);

		sutil::displayFps(frame_count++);


		{
			static const ImGuiWindowFlags window_flags =
				ImGuiWindowFlags_NoTitleBar |
				ImGuiWindowFlags_AlwaysAutoResize |
				ImGuiWindowFlags_NoMove |
				ImGuiWindowFlags_NoScrollbar;

			ImGui::SetNextWindowPos(ImVec2(2.0f, 40.0f));
			ImGui::Begin("controls", 0, window_flags);

		





		//	bool light_changed = false;
		//	bool sample_changed = false;

		//	float pos[3] = { LIGHT_POS.x,LIGHT_POS.y,LIGHT_POS.z };
		//	if (ImGui::SliderFloat3("light X,Y,Z\n ", pos, -4.0f, 4.0f, "%.2f")) {
		//		LIGHT_POS = make_float3(pos[0], pos[1], pos[2]);
		//		light_changed = true;

		//	}

		//	//int *depth = &RR_BEGIN_DEPTH;
		//	//if (ImGui::SliderInt("depth:\n", depth, 1, 10)) {
		//	//	context["rr_begin_depth"]->setUint(RR_BEGIN_DEPTH);
		//	//	FRAME_NUMBER = 0;
		//	//}

		//	//RR_BEGIN_DEPTH



			// Sample control
			int *sample = &SAMPLE_PIXEL;
			if (ImGui::SliderInt("sample:\n", sample, 1, 20)) {
				context["sqrt_num_samples"]->setUint(SAMPLE_PIXEL);
				FRAME_NUMBER = 0;
			}


			int *depth = &TRACE_DEPTH;

			if (ImGui::SliderInt("trace dpth:\n", depth, 1, 20)) {
				context["trace_depth"]->setUint(TRACE_DEPTH);
				FRAME_NUMBER = 0;
			}

		///*	float *sample = &SAMPLE_PIXEL;
		//	if (ImGui::SliderFloat("sample:\n", sample, 1, 10)) {
		//		context["sqrt_num_samples"]->setFloat(SAMPLE_PIXEL);
		//		FRAME_NUMBER = 0;
		//	}*/


	


			// Lerp control
			if (ImGui::Checkbox("lerping", &LERP_BOL)) {
				if (LERP_BOL) {
					context["lerp_bol"]->setInt(LERP_BOL);
				}
				else {
					context["lerp_bol"]->setInt(0);
				}
				FRAME_NUMBER = 0;
			}



			//// window control
			//if (ImGui::Checkbox("size", &window_128)) {
			//	if (window_128) {
			//		WIDTH = 128;
			//		HEIGHT = 128;
			//		glfwSetWindowSize(window, (int)WIDTH, (int)HEIGHT);
			//		glfwSetWindowSizeCallback(window, windowSizeCallback);
			//	}
			//	else {
			//		WIDTH = 256;
			//		HEIGHT = 256;
			//		glfwSetWindowSize(window, (int)WIDTH, (int)HEIGHT);
			//		glfwSetWindowSizeCallback(window, windowSizeCallback);
			//	}
			//	FRAME_NUMBER = 0;
			//}


		//	//light change
		//	if (light_changed) {
		//		ParallelogramLight light;
		//		light.corner = LIGHT_POS;
		//		light.v1 = make_float3(1.0f, 0.0f, 0.0f);
		//		light.v2 = make_float3(0.0f, 0.0f, 1.0f);
		//		light.normal = normalize(cross(light.v1, light.v2));
		//		light.emission = make_float3(15.0f, 15.0f, 5.0f);

		//		Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
		//		light_buffer->setFormat(RT_FORMAT_USER);
		//		light_buffer->setElementSize(sizeof(ParallelogramLight));
		//		light_buffer->setSize(1u);
		//		memcpy(light_buffer->map(), &light, sizeof(light));
		//		light_buffer->unmap();
		//		context["lights"]->setBuffer(light_buffer);

		//		FRAME_NUMBER = 0;
		//	}


			ImGui::End();
		}

		


		// imgui pops
		ImGui::PopStyleVar(3);

		// Render main window
		context["frame_number"]->setUint(FRAME_NUMBER++);
		context->launch(0, camera.width(), camera.height());
		sutil::displayBufferGL(outputbuffer);


		//// get noisy Data
		//if (FRAME_NUMBER == 1 || FRAME_NUMBER == 2) {
		//	//printf("%f:    \n", frame_number);
		//	std::string outputImage = "D:/Dataset/testing dataset/sponza/color" + std::string(SAMPLE_NAME) + "noise_" + std::to_string(noise_number) + ".ppm";
		//	displayBufferPPM(outputImage.c_str(), getOutputBuffer()->get(), false);
		//	outputImage = "D:/Dataset/testing dataset/sponza/normal/" + std::string(SAMPLE_NAME) + std::to_string(noise_number) + ".ppm";
		//	displayBufferPPM(outputImage.c_str(), getNormalBuffer()->get(), false);
		//	outputImage = "D:/Dataset/testing dataset/sponza/depth/" + std::string(SAMPLE_NAME) + std::to_string(noise_number) + ".ppm";
		//	displayBufferPPM(outputImage.c_str(), getdepthBuffer()->get(), false);
		//	std::cerr << "Saving noise frame to '" << outputImage << "'\n";




		//	context["sqrt_num_samples"]->setUint(20);
		//	FRAME_NUMBER = 0;
		//	context->launch(0, camera.width(), camera.height());
		//	outputImage = "D:/Dataset/testing dataset/sponza/ground truth/" + std::string(SAMPLE_NAME) + std::to_string(noise_number) + ".ppm";
		//	displayBufferPPM(outputImage.c_str(), getOutputBuffer()->get(), false);
		//	std::cerr << "Saving fine frame to '" << outputImage << "'\n";


		//	context["sqrt_num_samples"]->setUint(1);



		//	unsigned seed;
		//	seed = (unsigned)time(NULL); // 取得時間序列
		//	srand(seed); // 以時間序列當亂數種子
		//	noise_number++;
		//	float x = min.x + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.x - min.x)));
		//	float y = min.y + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.y - min.y)));
		//	float z = min.z + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.z - min.z)));
		//	CAMERA_POS = make_float3(x, y, z);
		//	//printf("%f,%f,%f\n", x, y, z);
		//	x = min.x + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.x - min.x)));
		//	y = min.y + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.y - min.y)));
		//	z = min.z + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.z - min.z)));
		//	CAMERA_LOOKAT = make_float3(x, y, z);
		//	x = min.x + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.x - min.x)));
		//	y = min.y + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.y - min.y)));
		//	z = min.z + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.z - min.z)));
		//	CAMERA_UP = make_float3(x, y, z);
		//
		//
		//	sutil::Camera camera(WIDTH, HEIGHT,
		//		&CAMERA_POS.x, &CAMERA_LOOKAT.x, &CAMERA_UP.x,
		//		context["eye"], context["U"], context["V"], context["W"]);
		//	setNormalMatrix(camera);
		//	//LERP_BOL = false;
		//	//context["lerp_bol"]->setInt(0);
		//	//camera.reset_lookat();
		//	FRAME_NUMBER = 0;
		//	context->launch(0, camera.width(), camera.height());
		//}



		////get 7frame Data
		//if (FRAME_NUMBER == 64) {
		//	
		//	std::string outputImage = "./data/dataset/reference/" + std::string(SAMPLE_NAME) + std::to_string(noise_number) + ".ppm";
		//	std::cerr << "Saving fine frame to '" << outputImage << "'\n";
		//	displayBufferPPM(outputImage.c_str(), getOutputBuffer()->get(), false);
		//	//outputImage = "./data/origin/n/" + std::string(SAMPLE_NAME) + std::to_string(image_number) + ".ppm";
		//	//displayBufferPPM(outputImage.c_str(), getNormalBuffer()->get(), false);
		//	//outputImage = "./data/origin/d/" + std::string(SAMPLE_NAME) + std::to_string(image_number) + ".ppm";
		//	//displayBufferPPM(outputImage.c_str(), getDepthBuffer()->get(), false);
		//	image_number++;
		//	//std::cerr << "fps: " << frame_number << "\n";
		//	//last_frame = FRAME_NUMBER;
		//	unsigned seed;
		//	seed = (unsigned)time(NULL); // 取得時間序列
		//	srand(seed); // 以時間序列當亂數種子
		//	noise_number++;
		//	float x = min.x + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.x - min.x)));
		//	float y = min.y + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.y - min.y)));
		//	float z = min.z + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.z - min.z)));
		//	CAMERA_POS = make_float3(x, y, z);
		//	//printf("%f,%f,%f\n", x, y, z);
		//	x = min.x + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.x - min.x)));
		//	y = min.y + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.y - min.y)));
		//	z = min.z + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.z - min.z)));
		//	CAMERA_LOOKAT = make_float3(x, y, z);
		//	x = min.x + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.x - min.x)));
		//	y = min.y + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.y - min.y)));
		//	z = min.z + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max.z - min.z)));
		//	CAMERA_UP = make_float3(x, y, z);
		//	sutil::Camera camera(WIDTH, HEIGHT,
		//		&CAMERA_POS.x, &CAMERA_LOOKAT.x, &CAMERA_UP.x,
		//		context["eye"], context["U"], context["V"], context["W"]);
		//	setNormalMatrix(camera);
		//	//LERP_BOL = false;
		//	//context["lerp_bol"]->setInt(0);
		//	//camera.reset_lookat();
		//	FRAME_NUMBER = 0;
		//	context->launch(0, camera.width(), camera.height());
		//	//setCameraPostition(camera);
		//}			
		//if(noise_number==100) {
		//	if (context)
		//		context->destroy();
		//	if (window)
		//		glfwDestroyWindow(window);
		//	glfwTerminate();
		//	exit(EXIT_SUCCESS);
		//}

		//if (image_number <= 430) {
		//	//printf("%d\n",FRAME_NUMBER);
		//	////get noisy Data
		//	//if (FRAME_NUMBER == 1||FRAME_NUMBER==2) {
		//	//	////printf("%f:    \n", frame_number);
		//	//	std::string outputImage = "./data/dataset/noise/color/" + std::string(SAMPLE_NAME) + "noise_" + std::to_string(noise_number) + ".ppm";
		//	//	displayBufferPPM(outputImage.c_str(), getOutputBuffer()->get(), false);
		//	//	outputImage = "./data/dataset/noise/normal/" + std::string(SAMPLE_NAME) + std::to_string(image_number) + ".ppm";
		//	//	displayBufferPPM(outputImage.c_str(), getNormalBuffer()->get(), false);
		//	//	outputImage = "./data/dataset/noise/depth/" + std::string(SAMPLE_NAME) + std::to_string(image_number) + ".ppm";
		//	//	displayBufferPPM(outputImage.c_str(), getdepthBuffer()->get(), false);
		//	//	std::cerr << "Saving noise frame to '" << outputImage << "'\n";
		//
		//	//	noise_number++;
		//	//	image_number++;
		//
		//	//	FRAME_NUMBER = 0;
		//	//	context->launch(0, camera.width(), camera.height());
		//	//	setCameraPostition(camera);
		//	//	setNormalMatrix(camera);
		//
		//	//}
		//	
		//	////if (FRAME_NUMBER == 64) {
		//	//if (FRAME_NUMBER == 1 || FRAME_NUMBER == 2) {
		//	//	
		//	//	std::string outputImage = "./data/dataset/reference/" + std::string(SAMPLE_NAME) + std::to_string(noise_number) + ".ppm";
		//	//	std::cerr << "Saving fine frame to '" << outputImage << "'\n";
		//	//	displayBufferPPM(outputImage.c_str(), getOutputBuffer()->get(), false);
		//
		//	//	//outputImage = "./data/origin/n/" + std::string(SAMPLE_NAME) + std::to_string(image_number) + ".ppm";
		//	//	//displayBufferPPM(outputImage.c_str(), getNormalBuffer()->get(), false);
		//	//	//outputImage = "./data/origin/d/" + std::string(SAMPLE_NAME) + std::to_string(image_number) + ".ppm";
		//	//	//displayBufferPPM(outputImage.c_str(), getDepthBuffer()->get(), false);
		//
		//	//	noise_number++;
		//	//	image_number++;
		//
		//	//	FRAME_NUMBER = 0;
		//	//	context->launch(0, camera.width(), camera.height());
		//	//	setCameraPostition(camera);
		//	//	setNormalMatrix(camera);
		//	//}
		//}
		//else {
		//	if (context)
		//		context->destroy();
		//	if (window)
		//		glfwDestroyWindow(window);
		//	glfwTerminate();
		//	exit(EXIT_SUCCESS);
		//}

		//getTrainingData(noise_number, camera, max,min);
		// Render gui over it
		ImGui::Render();
		ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
	}
	destroyContext();
	glfwDestroyWindow(window);
	glfwTerminate();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

//#include "dnn.cu"

int main()
{
	//main2();
	try {

		GLFWwindow* window = glfwInitialize();

		createContext();
		createGeometry();
		context->validate();
		float3 camera_u, camera_v, camera_w;

		sutil::Camera camera(WIDTH, HEIGHT,
			&CAMERA_POS.x, &CAMERA_LOOKAT.x, &CAMERA_UP.x,
			context["eye"], context["U"], context["V"], context["W"]);

		setNormalMatrix(camera);
		

		glfwRun(window, camera, getOutputBuffer());
		//glfwRun(window, camera, getNormalBuffer());
		
		
		//system("pause");
		return 0;
	

	}	
	SUTIL_CATCH(context->get())
}
