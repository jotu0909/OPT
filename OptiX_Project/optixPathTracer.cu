#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "optixPathTracer.h"
#include "include\random.h"
#include "include\helpers.h"
#include <optix_device.h>

using namespace optix;

struct PerRayData_pathtrace
{
	float3 result;
	float3 normal;
	float3 radiance;
	float3 attenuation;
	float3 origin;
	float3 direction;
	unsigned int seed;
	int depth;
	int countEmitted;
	int done;
	float depthMap;
	bool alpha;

};

struct PerRayData_pathtrace_shadow
{
	bool inShadow;
};

// Scene wide variables
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );


//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );

rtDeclareVariable(unsigned int, frame_number, , );
rtDeclareVariable(unsigned int, sqrt_num_samples, , );
//rtDeclareVariable(float, sqrt_num_samples, , );
rtDeclareVariable(unsigned int, rr_begin_depth, , );
rtDeclareVariable(unsigned int, pathtrace_ray_type, , );
rtDeclareVariable(unsigned int, pathtrace_shadow_ray_type, , );


rtDeclareVariable(float3, emission_color, , );
rtDeclareVariable(float3, diffuse_color, , );
rtDeclareVariable(float3, specular_color, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

rtDeclareVariable(float3, bg_color, , );
rtDeclareVariable(float3, bad_color, , );

rtDeclareVariable(float3, Ka, , );
rtDeclareVariable(unsigned int, trace_depth, , );


rtTextureSampler<float4, 2> Kd_map;
rtTextureSampler<float4, 2> d_map;
rtTextureSampler<float4, 2> Ks_map;
rtTextureSampler<float4, 2> bump_map;

rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(Matrix3x3, normal_matrix, , );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>			     input_normal_buffer;
rtBuffer<float4, 2>			     input_depth_buffer;
rtBuffer<ParallelogramLight>     lights;

rtDeclareVariable(int, lerp_bol, , );


RT_PROGRAM void pathtrace_camera()
{
	//printf("%f, %f, %f\n", Ka.x, Ka.y, Ka.z);
		size_t2 screen = output_buffer.size();

		float2 inv_screen = 1.0f / make_float2(screen) * 2.f;
		float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

		float2 jitter_scale = inv_screen / sqrt_num_samples;

		unsigned int samples_per_pixel = sqrt_num_samples*sqrt_num_samples;
		float3 result = make_float3(0.0f);
		float3 normal = make_float3(0.0f);
		float depth_map = 0.0f;
		//bool alpha = false;

		unsigned int seed = tea<16>(screen.x*launch_index.y + launch_index.x, frame_number);

		do
		{
			//
			// Sample pixel using jittering
			//
			unsigned int x = samples_per_pixel%sqrt_num_samples;
			//unsigned int x = 0;
			unsigned int y = samples_per_pixel / sqrt_num_samples;
			float2 jitter = make_float2(x /*- rnd(seed)*/, y /*- rnd(seed)*/);
			float2 d = pixel + jitter*jitter_scale;
			float3 ray_origin = eye;
			float3 ray_direction = normalize(d.x*U + d.y*V + W);

			// Initialze per-ray data
			PerRayData_pathtrace prd;
			prd.result = make_float3(0.f);
			prd.attenuation = make_float3(1.f);
			prd.countEmitted = true;
			prd.done = false;
			prd.seed = seed;
			prd.depth = 0;
			prd.depthMap = 0.0f;
			prd.normal = make_float3(0.0);
			prd.alpha = false;

			// Each iteration is a segment of the ray path.  The closest hit will
			// return new segments to be traced here.
			for (;;)
			{
				Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
				rtTrace(top_object, ray, prd);
				
				if (prd.done)
				{
					// We have hit the background or a luminaire
						prd.result += prd.radiance * prd.attenuation; 
					break;
				}
				
				// Russian roulette termination 
				if (prd.depth >= rr_begin_depth)//
				{
					//if(prd.depth>trace_depth-1){
					//	//printf("%d\n",prd.depth);
					//	prd.result += prd.radiance * prd.attenuation;
					//	break;
					//	//prd.done = true;
					//}
					float pcont = fmaxf(prd.attenuation);
					if (rnd(prd.seed) >= pcont)
						break;
					prd.attenuation /= pcont;			
				}
					
				//if (prd.alpha) {
				//	prd.depth++;
				//	prd.alpha = false;
				//}


					prd.depth++;
				
				//// Without Russian roulette termination 
				//prd.done = true;
				
				prd.result += prd.radiance * prd.attenuation;

				// Update ray data for the next path segment
				ray_origin = prd.origin;
				ray_direction = prd.direction;
			}

			result += prd.result;
			float3 normal_eyespace = (length(prd.normal) > 0.f) ? normalize(normal_matrix * prd.normal) : make_float3(0., 0., 0.);
			normal += normal_eyespace;
			seed = prd.seed;
			depth_map += prd.depthMap;
		} while (--samples_per_pixel);

		//
		// Update the output buffer
		//
		unsigned int spp = sqrt_num_samples*sqrt_num_samples;
		float3 pixel_color = result / spp;
		float3 pixel_normal = normal / (spp);
		float pixel_depth =1- depth_map /6;


		if (frame_number > 1)
		{
			float a = 1.0f / (float)frame_number;
			float3 old_color = make_float3(output_buffer[launch_index]);
			float3 old_normal = make_float3(input_normal_buffer[launch_index]);
			float3 old_depth = make_float3(input_depth_buffer[launch_index]);

			if (lerp_bol) {
				output_buffer[launch_index] = make_float4(lerp(old_color, pixel_color, a), 1.0f);
				// this is not strictly a correct accumulation of normals, but it will do for this sample
				float3 accum_normal = lerp(old_normal, pixel_normal, a);
				input_normal_buffer[launch_index] = make_float4((length(accum_normal) > 0.f) ? normalize(accum_normal) : pixel_normal, 1.0f);
				input_depth_buffer[launch_index] = make_float4(lerp(old_depth, make_float3(pixel_depth), a), 1.0f);

			}
			else {
				output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
				input_normal_buffer[launch_index] = make_float4(pixel_normal, 1.0f);
				input_depth_buffer[launch_index] = make_float4(make_float3(pixel_depth), 1.0f);

			}
		}
		else
		{
			output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
			input_normal_buffer[launch_index] = make_float4(pixel_normal, 1.0f);
			input_depth_buffer[launch_index] = make_float4(make_float3(pixel_depth), 1.0f);


		}

	
}


//-----------------------------------------------------------------------------
//
//  Emissive surface closest-hit
//
//-----------------------------------------------------------------------------



RT_PROGRAM void diffuseEmitter()
{

	//current_prd.radiance = current_prd.countEmitted ? emission_color : make_float3(1.f);
	current_prd.done = true;

	//// TODO: Find out what the albedo buffer should really have. For now just set to white for 
	//// light sources.
	//if (current_prd.depth == 0 && length(current_prd.normal) == 0)
	//{

	//	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	//	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	//	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	//	current_prd.normal = ffnormal;
	//	current_prd.depthMap =t_hit;
	//}


}


//-----------------------------------------------------------------------------
//
//  Lambertian surface closest-hit
//
//-----------------------------------------------------------------------------

RT_PROGRAM void alpha_texture_hit()
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 hitpoint = ray.origin + t_hit * ray.direction;
	//float distance = length(hitpoint - ray.origin);
	//current_prd.alph = false;
	//
	// Generate a reflection ray.  This will be traced back in ray-gen.
	//
	float3 direction = ray.direction;
	current_prd.origin = hitpoint;


	const float3 Kd_val = make_float3(tex2D(Kd_map, texcoord.x, texcoord.y));
	const float3 Ks_val = make_float3(tex2D(Ks_map, texcoord.x, texcoord.y));
	const float3 d_val = make_float3(tex2D(d_map, texcoord.x, texcoord.y));
	const float3 bump_val2 = make_float3(tex2D(bump_map, texcoord.x, texcoord.y));
	const float3 bump_val = make_float3(bump_val2.z, bump_val2.z, bump_val2.z);
	// Initialze per-ray data


	// NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
	// with cosine density.
	if (d_val.x == 0) {// hit leaves texture ,trace again

		
		Ray ray2 = make_Ray(hitpoint, direction, 0u, scene_epsilon, RT_DEFAULT_MAX);
		rtTrace(top_object, ray2, current_prd);
	
		//current_prd.alpha = true;
		if (current_prd.depth == 0 )
		{				
			current_prd.depthMap += t_hit;			
		}
		
		
	}
	else
	{		
		if (current_prd.depth == 0)
		{
			if (!current_prd.alpha) 
			{
				current_prd.normal = ffnormal;
			}
			current_prd.depthMap = t_hit;
		}



		float z1 = rnd(current_prd.seed);
		float z2 = rnd(current_prd.seed);
		float3 p;
		cosine_sample_hemisphere(z1, z2, p);
		optix::Onb onb(ffnormal);
		onb.inverse_transform(p);
		current_prd.direction = p;


		current_prd.attenuation = current_prd.attenuation/**Ka*/*Kd_val *Ks_val*bump_val;
		current_prd.countEmitted = false;
		
		// Next event estimation (compute direct lighting).
		//
		unsigned int num_lights = lights.size();
		float3 result =  Kd_val*Ka*Ks_val*bump_val;//make_float3(0.0);

		for (int i = 0; i < num_lights; ++i)
		{
			// Choose random point on light
			ParallelogramLight light = lights[i];
			//const float z1 = rnd(current_prd.seed);
			//const float z2 = rnd(current_prd.seed);
			const float3 light_pos = light.corner;//+light.v1 * z1 + light.v2 * z2;

			// Calculate properties of light sample (for area based pdf)
			const float  Ldist = length(light_pos - hitpoint);
			const float3 L = normalize(light_pos - hitpoint);
			const float  nDl = dot(ffnormal, L);
			//const float  LnDl = dot(light.normal, L);
			
			// cast shadow ray
			if (nDl > 0.0f /*&& LnDl > 0.0f*/)
			{
				PerRayData_pathtrace_shadow shadow_prd;
				shadow_prd.inShadow = false;
				// Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
				Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
				rtTrace(top_object, shadow_ray, shadow_prd);

				if (!shadow_prd.inShadow)
				{
					//const float A = length(cross(light.v1, light.v2));
					// convert area based pdf to solid angle
					const float weight = nDl /** LnDl * A *// (M_PIf * Ldist * Ldist);
					result += light.emission * weight ;
				}
			}
		}

		current_prd.radiance = result;
	}
}


RT_PROGRAM void texture_hit()
{

	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);


	float3 hitpoint = ray.origin + t_hit * ray.direction;
	

	if (current_prd.depth == 0 )
	{
		if (!current_prd.alpha) {
			current_prd.normal = ffnormal;
		}
		current_prd.depthMap = t_hit;
		
	}
	
	//
	// Generate a reflection ray.  This will be traced back in ray-gen.
	//
	current_prd.origin = hitpoint;

	float z1 = rnd(current_prd.seed);
	float z2 = rnd(current_prd.seed);
	float3 p;
	cosine_sample_hemisphere(z1, z2, p);
	optix::Onb onb(ffnormal);
	onb.inverse_transform(p);
	current_prd.direction = p;

	const float3 Kd_val = make_float3(tex2D(Kd_map, texcoord.x, texcoord.y));
	const float3 Ks_val = make_float3(tex2D(Ks_map, texcoord.x, texcoord.y));
	const float3 bump_val2 = make_float3(tex2D(bump_map, texcoord.x, texcoord.y));
	const float3 bump_val = make_float3(bump_val2.z, bump_val2.z, bump_val2.z);


	// NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
	// with cosine density.
	current_prd.attenuation = current_prd.attenuation*Kd_val*Ks_val*bump_val;
	current_prd.countEmitted = false;

	//
	// Next event estimation (compute direct lighting).
	//
	unsigned int num_lights = lights.size();
	float3 result =  Kd_val*Ka*Ks_val*bump_val;//make_float3(0.0);

	for (int i = 0; i < num_lights; ++i)
	{
		// Choose random point on light
		ParallelogramLight light = lights[i];
		const float z1 = rnd(current_prd.seed);
		const float z2 = rnd(current_prd.seed);
		const float3 light_pos = light.corner;// +light.v1 * z1 + light.v2 * z2;//

		// Calculate properties of light sample (for area based pdf)
		const float  Ldist = length(light_pos - hitpoint);
		const float3 L = normalize(light_pos - hitpoint);
		const float  nDl = dot(ffnormal, L);
		//const float  LnDl = dot(light.normal, L);

		// cast shadow ray
		if (nDl > 0.0f/* && LnDl > 0.0f*/)
		{
			PerRayData_pathtrace_shadow shadow_prd;
			shadow_prd.inShadow = false;
			// Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
			Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
			rtTrace(top_object, shadow_ray, shadow_prd);

			if (!shadow_prd.inShadow)
			{
				//const float A = length(cross(light.v1, light.v2));
				//convert area based pdf to solid angle
				const float weight = nDl /** LnDl * A *// (M_PIf * Ldist * Ldist);
				result += light.emission * weight ;
			}
			
		}
	}

	current_prd.radiance = result;
}

RT_PROGRAM void mirror()
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 hitpoint = ray.origin + t_hit * ray.direction;


	if (current_prd.depth == 0 )
	{
		if (!current_prd.alpha) {
			current_prd.normal = ffnormal;
		}
		current_prd.depthMap = t_hit;
	}

	//
	// Generate a reflection ray.  This will be traced back in ray-gen.
	//
	current_prd.origin = hitpoint;


	float3 p;
	p = reflect(ray.direction, ffnormal);
	//optix::Onb onb(ffnormal);
	//onb.inverse_transform(p);
	current_prd.direction = p;

	// NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
	// with cosine density.
	current_prd.attenuation = current_prd.attenuation * diffuse_color+(specular_color*0.3);//
	current_prd.countEmitted = false;

	// Next event estimation (compute direct lighting).
	//
	unsigned int num_lights = lights.size();
	float3 result = diffuse_color*Ka + specular_color*0.3;

	for (int i = 0; i < num_lights; ++i)
	{
		// Choose random point on light
		ParallelogramLight light = lights[i];
		const float z1 = rnd(current_prd.seed);
		const float z2 = rnd(current_prd.seed);
		const float3 light_pos = light.corner +light.v1 * z1 + light.v2 * z2;//

											  // Calculate properties of light sample (for area based pdf)
		const float  Ldist = length(light_pos - hitpoint);
		const float3 L = normalize(light_pos - hitpoint);
		const float  nDl = dot(ffnormal, L);		// the normal of hitpoint's normal dot L to hitpoint.  
		const float  LnDl = dot(light.normal, L);	// the normal of L's		normal dot L to hitpoint. 

													// cast shadow ray
		if (nDl > 0.0f && LnDl > 0.0f)
		{
			PerRayData_pathtrace_shadow shadow_prd;
			shadow_prd.inShadow = false;
			// Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
			Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
			rtTrace(top_object, shadow_ray, shadow_prd);

			if (!shadow_prd.inShadow)
			{

				const float A = length(cross(light.v1, light.v2));
				// convert area based pdf to solid angle
				const float weight = nDl  *LnDl * A  / (M_PIf * Ldist * Ldist);
				result += light.emission * weight;
			}
		}
	}

	current_prd.radiance = result;
}



RT_PROGRAM void diffuse()
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 hitpoint = ray.origin + t_hit * ray.direction;
	

	if (current_prd.depth == 0 )
	{
		if (!current_prd.alpha) {
			current_prd.normal = ffnormal;
		}
		current_prd.depthMap = t_hit;
	}
	//
	// Generate a reflection ray.  This will be traced back in ray-gen.
	//
	current_prd.origin = hitpoint;

	float z1 = rnd(current_prd.seed);
	float z2 = rnd(current_prd.seed);
	float3 p;
	cosine_sample_hemisphere(z1, z2, p);
	optix::Onb onb(ffnormal);
	onb.inverse_transform(p);
	current_prd.direction = p;

	// NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
	// with cosine density.
	current_prd.attenuation = current_prd.attenuation * diffuse_color+(specular_color*0.3);//
	current_prd.countEmitted = false;

	// Next event estimation (compute direct lighting).
	//
	unsigned int num_lights = lights.size();
	float3 result = diffuse_color *Ka+specular_color*0.3; //make_float3(0.0);//;//;// 

	for (int i = 0; i < num_lights; ++i)
	{
		// Choose random point on light
		ParallelogramLight light = lights[i];
		//const float z1 = rnd(current_prd.seed);
		//const float z2 = rnd(current_prd.seed);
		const float3 light_pos = light.corner;// +light.v1 * z1 + light.v2 * z2;

		// Calculate properties of light sample (for area based pdf)
		const float  Ldist = length(light_pos - hitpoint);
		const float3 L = normalize(light_pos - hitpoint);
		const float  nDl = dot(ffnormal, L);		// the normal of hitpoint's normal dot L to hitpoint.  
		//const float  LnDl = dot(light.normal, L);	// the normal of L's		normal dot L to hitpoint. 

		// cast shadow ray
		if (nDl > 0.0f/* && LnDl > 0.0f*/)
		{
			PerRayData_pathtrace_shadow shadow_prd;
			shadow_prd.inShadow = false;
			// Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
			Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
			rtTrace(top_object, shadow_ray, shadow_prd);

			if (!shadow_prd.inShadow)
			{

				//const float A = length(cross(light.v1, light.v2));
				// convert area based pdf to solid angle
				const float weight = nDl /* *LnDl * A*/ / (M_PIf * Ldist * Ldist);
				result += light.emission * weight ;
			}
		}
	}

	current_prd.radiance = result;
}


//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------


RT_PROGRAM void shadow()
{

	const float3 d_val = make_float3(tex2D(d_map, texcoord.x, texcoord.y));
	if (d_val.x != 0) {
		current_prd_shadow.inShadow = true;
		rtTerminateRay();
	}

}



//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
	
	printf("%d\n", rtGetExceptionCode());
	output_buffer[launch_index] = make_float4(bad_color, 1.0f);
	input_normal_buffer[launch_index] = make_float4(bad_color, 1.0f);
	input_depth_buffer[launch_index] = make_float4(bad_color, 1.0f);
}


//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------


RT_PROGRAM void miss()
{
	
	current_prd.radiance = bg_color;
	current_prd.done = true;

	// TODO: Find out what the albedo buffer should really have. For now just set to black for misses.
	if (current_prd.depth == 0)
	{
		current_prd.normal = make_float3(0, 0, 0);
		current_prd.depthMap = 0;
	}

}

rtTextureSampler<float4, 2>		envmap;
RT_PROGRAM void envmap_miss()
{
	
	float theta = atan2f(ray.direction.x, ray.direction.z);
	float phi = M_PIf * 0.5f - acosf(ray.direction.y);
	float u = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v = 0.5f * (1.0f + sin(phi));
	float3 result = make_float3(tex2D(envmap, u, v));
	
	current_prd.radiance = result;
	//current_prd.attenuation =make_float3(0.0);
		//make_float3(tex2D(envmap, u, v));
	current_prd.done = true;

	if (current_prd.depth == 0)
	{
		current_prd.normal = make_float3(0, 0, 0);
		current_prd.depthMap = 0;
	}

}