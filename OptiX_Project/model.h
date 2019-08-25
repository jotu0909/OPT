#pragma once
#include <string>
#include <vector>
#include <map>
#include <assert.h>
#include <optixpp_namespace.h>
#include "sutil\sutil.h"
#include "sutil\Mesh.h"

#define TINYOBJ_SSCANF_BUFFER_SIZE  (4096)

struct material_t {

	std::string name;
	std::string emssion_texture;			// map_Ke
	std::string ambient_texname;            // map_Ka
	std::string diffuse_texname;            // map_Kd
	std::string specular_texname;           // map_Ks
	std::string specular_highlight_texname; // map_Ns
	std::string bump_texname;               // map_bump, bump
	std::string displacement_texname;       // disp
	std::string alpha_texname;              // map_d

};


//static void InitMaterial(material_t &material);

static inline bool isSpace(const char c) { return (c == ' ') || (c == '\t'); }

void LoadMtl(
	std::map<std::string, int> &material_map,
	std::vector<material_t> &materials,
	std::istream &inStream
);

std::vector<material_t> setuptexture(
	std::map<std::string, int> &material_map,
	std::vector<material_t> &materials
);



// Utilities for translating Mesh data to OptiX buffers.  These are copied and pasted from sutil.

struct MeshBuffers {

	optix::Buffer tri_indices;				//triangle vertex index
	optix::Buffer mat_indices;				//material index
	optix::Buffer positions;				//vertex
	optix::Buffer normals;					//normals
	optix::Buffer texcoords;				//texture

};


void setupMeshLoaderInputs(
	optix::Context            context,
	MeshBuffers&              buffers,
	Mesh&                     mesh);

void unmap(MeshBuffers& buffers, Mesh& mesh);


