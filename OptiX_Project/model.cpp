#include "model.h"


static void InitMaterial(material_t &material) {
	material.name = "";
	material.ambient_texname = "";
	material.diffuse_texname = "";
	material.specular_texname = "";
	material.specular_highlight_texname = "";
	material.bump_texname = "";
	material.displacement_texname = "";
	material.alpha_texname = "";

}

void LoadMtl(
	std::map<std::string, int> &material_map,
	std::vector<material_t> &materials,
	std::istream &inStream
) {

	material_t material;
	InitMaterial(material);

	size_t maxchars = 8192;             // Alloc enough size.
	std::vector<char> buf(maxchars); // Alloc enough size.
	while (inStream.peek() != -1) {
		inStream.getline(&buf[0], static_cast<std::streamsize>(maxchars));

		std::string linebuf(&buf[0]);

		// Trim newline '\r\n' or '\n'
		if (linebuf.size() > 0) {
			if (linebuf[linebuf.size() - 1] == '\n')
				linebuf.erase(linebuf.size() - 1);
		}
		if (linebuf.size() > 0) {
			if (linebuf[linebuf.size() - 1] == '\r')
				linebuf.erase(linebuf.size() - 1);
		}

		// Skip if empty line.
		if (linebuf.empty()) {
			continue;
		}

		// Skip leading space.
		const char *token = linebuf.c_str();
		token += strspn(token, " \t");

		assert(token);
		if (token[0] == '\0')
			continue; // empty line

		if (token[0] == '#')
			continue; // comment line

					  // new mtl
		if ((0 == strncmp(token, "newmtl", 6)) && isSpace((token[6]))) {
			// flush previous material.
			if (!material.name.empty()) {
				material_map.insert(
					std::pair<std::string, int>(material.name, static_cast<int>(materials.size())));
				materials.push_back(material);
			}

			// initial temporary material
			InitMaterial(material);

			// set new mtl name
			char namebuf[TINYOBJ_SSCANF_BUFFER_SIZE];
			token += 7;
#ifdef _MSC_VER
			sscanf_s(token, "%s", namebuf, (unsigned)_countof(namebuf));
#else
			sscanf(token, "%s", namebuf);
#endif
			material.name = namebuf;
			continue;
		}

		//  emission texture
		if ((0 == strncmp(token, "map_Ke", 6)) && isSpace(token[6])) {
			token += 7;
			material.emssion_texture = token;
			//printf("Yes");
			continue;
		}

		// ambient texture
		if ((0 == strncmp(token, "map_Ka", 6)) && isSpace(token[6])) {
			token += 7;
			material.ambient_texname = token;
			continue;
		}

		// diffuse texture
		if ((0 == strncmp(token, "map_Kd", 6)) && isSpace(token[6])) {
			token += 7;
			material.diffuse_texname = token;
			continue;
		}

		// specular texture
		if ((0 == strncmp(token, "map_Ks", 6)) && isSpace(token[6])) {
			token += 7;
			material.specular_texname = token;
			continue;
		}

		// specular highlight texture
		if ((0 == strncmp(token, "map_Ns", 6)) && isSpace(token[6])) {
			token += 7;
			material.specular_highlight_texname = token;
			continue;
		}

		// bump texture
		if ((0 == strncmp(token, "map_Bump", 8)) && isSpace(token[8])) {
			token += 9;
			material.bump_texname = token;
			continue;
		}

		// alpha texture
		if ((0 == strncmp(token, "map_d", 5)) && isSpace(token[5])) {
			token += 6;
			material.alpha_texname = token;
			continue;
		}

		// bump texture
		if ((0 == strncmp(token, "bump", 4)) && isSpace(token[4])) {
			token += 5;
			material.bump_texname = token;
			continue;
		}

		// displacement texture
		if ((0 == strncmp(token, "disp", 4)) && isSpace(token[4])) {
			token += 5;
			material.displacement_texname = token;
			continue;
		}


		// flush last material.
		material_map.insert(
			std::pair<std::string, int>(material.name, static_cast<int>(materials.size())));
		materials.push_back(material);


	}
}



std::vector<material_t> setuptexture(
	std::map<std::string, int> &material_map,
	std::vector<material_t> &materials
) {

	std::vector<material_t> vec_texture;
	vec_texture.resize(material_map.size());

	std::string name = materials[0].name;
	int j = 0;
	for (int i = material_map[name]; i < materials.size(); i++)
	{

		if (name == materials[i].name) {
			vec_texture[j] = materials[i];
		}
		else {
			j++;
			name = materials[i].name;
		}

	}
	return vec_texture;
}


void setupMeshLoaderInputs(
	optix::Context            context,
	MeshBuffers&              buffers,
	Mesh&                     mesh
) {
	buffers.tri_indices = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, mesh.num_triangles);
	buffers.mat_indices = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, mesh.num_triangles);
	buffers.positions	= context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, mesh.num_vertices);
	buffers.normals		= context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3,
							mesh.has_normals ? mesh.num_vertices : 0);
	buffers.texcoords	= context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2,
							mesh.has_texcoords ? mesh.num_vertices : 0);

	mesh.tri_indices = reinterpret_cast<int32_t*>(buffers.tri_indices->map());
	mesh.mat_indices = reinterpret_cast<int32_t*>(buffers.mat_indices->map());
	mesh.positions	 = reinterpret_cast<float*>	 (buffers.positions->map());
	mesh.normals	 = reinterpret_cast<float*>  (mesh.has_normals ? buffers.normals->map() : 0);
	mesh.texcoords   = reinterpret_cast<float*>  (mesh.has_texcoords ? buffers.texcoords->map() : 0);

	mesh.mat_params = new MaterialParams[mesh.num_materials];
}

void unmap(MeshBuffers& buffers, Mesh& mesh)
{
	buffers.tri_indices->unmap();
	buffers.mat_indices->unmap();
	buffers.positions->unmap();
	if (mesh.has_normals)
		buffers.normals->unmap();
	if (mesh.has_texcoords)
		buffers.texcoords->unmap();

	mesh.tri_indices = 0;
	mesh.mat_indices = 0;
	mesh.positions = 0;
	mesh.normals = 0;
	mesh.texcoords = 0;


	delete[] mesh.mat_params;
	mesh.mat_params = 0;
}