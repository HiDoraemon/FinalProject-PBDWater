#ifndef SCENE_H
#define SCENE_H

#include "glm/glm.hpp"
#include "utilities.h"
#include <vector>
#include "sceneStructs.h"
#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;

class scene{
private:
    ifstream fp_in;
    int loadObject(string objectid);
public:
    scene(string filename);
    ~scene();

    vector<geom> objects;
	int numCubes;
	int numSpheres;
};

#endif