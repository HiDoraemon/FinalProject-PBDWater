#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>

#define THRESHOLD .05

//Some forward declarations
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ float boxIntersectionTest(staticGeom sphere, glm::vec3 p_pos, glm::vec3& intersectionPoint, glm::vec3& normal, bool& inside);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, glm::vec3 p_pos, glm::vec3& intersectionPoint, glm::vec3& normal, bool& inside);

//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, glm::vec3 p_pos, glm::vec3& intersectionPoint, glm::vec3& normal, bool& inside){
	
	return -1.0;
}

//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, glm::vec3 p_pos, glm::vec3& intersectionPoint, glm::vec3& normal, bool& inside){
	float radius = .5;
	glm::vec3 obj_space_p_pos = multiplyMV(sphere.inverseTransform, glm::vec4(p_pos,1.0f));
	float distance = glm::length(obj_space_p_pos);

	if(distance <= radius){

	}else if(distance <= radius + THRESHOLD){

	}

}

#endif
