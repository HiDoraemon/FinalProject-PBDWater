#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>

#define THRESHOLD .0005f

//Some forward declarations
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ float boxIntersectionTest(staticGeom sphere, glm::vec3 p_pos, glm::vec3& intersectionPoint, glm::vec3& normal, bool& inside);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, glm::vec3 p_pos, glm::vec3& intersectionPoint, glm::vec3& normal, bool& inside);

//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, glm::vec3 p_pos, glm::vec3& intersectionPoint, glm::vec3& normal, bool& inside){
	
	return -1.0;
}

//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ bool sphereIntersectionTest(staticGeom sphere, glm::vec3 p_pos, glm::vec3& intersectionPoint, glm::vec3& normal){
	//the way i draw the visualize the sphere is with the radius as 1
	float radius = 1.0f;
	glm::vec3 obj_space_p_pos = multiplyMV(sphere.inverseTransform, glm::vec4(p_pos,1.0f));
	float distance = glm::length(obj_space_p_pos);

	glm::vec3 n = obj_space_p_pos*(1.0f/glm::length(obj_space_p_pos));

	/*if(distance <= radius){
		glm::vec3 obj_intersectionPoint = n*(radius+THRESHOLD);
		glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));
		glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(obj_intersectionPoint,1.0f));
		normal = glm::normalize(realIntersectionPoint-realOrigin);
		intersectionPoint = realIntersectionPoint;
		return true;
	}else*/
	if(distance < radius + THRESHOLD){
		glm::vec3 obj_intersectionPoint = n*(radius+THRESHOLD);
		glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));
		glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(obj_intersectionPoint,1.0f));
		normal = glm::normalize(realIntersectionPoint-realOrigin);
		intersectionPoint = realIntersectionPoint;
		return true;
	}
	
	return false;
}

__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}


#endif
