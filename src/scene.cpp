#include <iostream>
#include "scene.h"
#include <cstring>

scene::scene(string filename){
	numCubes = 0;
	numSpheres = 0;
	cout << "Reading scene from " << filename << " ..." << endl;
	cout << " " << endl;
	char* fname = (char*)filename.c_str();
	fp_in.open(fname);
	if(fp_in.is_open()){
		while(fp_in.good()){
			string line;
            utilityCore::safeGetline(fp_in,line);
			if(!line.empty()){
				vector<string> tokens = utilityCore::tokenizeString(line);
				if(strcmp(tokens[0].c_str(), "OBJECT")==0){
				    loadObject(tokens[1]);
				    cout << " " << endl;
				}
			}
		}
	}
}

int scene::loadObject(string objectid){
    int id = atoi(objectid.c_str());
    if(id!=objects.size()){
        cout << "ERROR: OBJECT ID does not match expected number of objects" << endl;
        return -1;
    }else{
        cout << "Loading Object " << id << "..." << endl;
        geom newObject;
        string line;
        
        //load object type 
        utilityCore::safeGetline(fp_in,line);
        if (!line.empty() && fp_in.good()){
            if(strcmp(line.c_str(), "sphere")==0){
                cout << "Creating new sphere..." << endl;
				numSpheres++;
				newObject.type = SPHERE;
            }else if(strcmp(line.c_str(), "cube")==0){
                cout << "Creating new cube..." << endl;
				numCubes++;
				newObject.type = CUBE;
            }else{
				string objline = line;
                string name;
                string extension;
                istringstream liness(objline);
                getline(liness, name, '.');
                getline(liness, extension, '.');
                if(strcmp(extension.c_str(), "obj")==0){
                    cout << "Creating new mesh..." << endl;
                    cout << "Reading mesh from " << line << "... " << endl;
		    		newObject.type = MESH;
                }else{
                    cout << "ERROR: " << line << " is not a valid object type!" << endl;
                    return -1;
                }
            }
        }
       
	//link material
    utilityCore::safeGetline(fp_in,line);
	if(!line.empty() && fp_in.good()){
	    vector<string> tokens = utilityCore::tokenizeString(line);
	}
	//load frames
    int frameCount = 0;
    utilityCore::safeGetline(fp_in,line);
	vector<glm::vec3> translations;
	vector<glm::vec3> scales;
	vector<glm::vec3> rotations;
    while (!line.empty() && fp_in.good()){
	    
	    //check frame number
	    vector<string> tokens = utilityCore::tokenizeString(line);
        if(strcmp(tokens[0].c_str(), "frame")!=0 || atoi(tokens[1].c_str())!=frameCount){
            cout << "ERROR: Incorrect frame count!" << endl;
            return -1;
        }
	    
	    //load tranformations
	    for(int i=0; i<3; i++){
            glm::vec3 translation; glm::vec3 rotation; glm::vec3 scale;
            utilityCore::safeGetline(fp_in,line);
            tokens = utilityCore::tokenizeString(line);
            if(strcmp(tokens[0].c_str(), "TRANS")==0){
                translations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }else if(strcmp(tokens[0].c_str(), "ROTAT")==0){
                rotations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }else if(strcmp(tokens[0].c_str(), "SCALE")==0){
                scales.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }
	    }
	    
	    frameCount++;
        utilityCore::safeGetline(fp_in,line);
	}
	
	//move frames into CUDA readable arrays
	newObject.translations = new glm::vec3[frameCount];
	newObject.rotations = new glm::vec3[frameCount];
	newObject.scales = new glm::vec3[frameCount];
	newObject.transforms = new cudaMat4[frameCount];
	newObject.inverseTransforms = new cudaMat4[frameCount];
	for(int i=0; i<frameCount; i++){
		newObject.translations[i] = translations[i];
		newObject.rotations[i] = rotations[i];
		newObject.scales[i] = scales[i];
		glm::mat4 transform = utilityCore::buildTransformationMatrix(translations[i], rotations[i], scales[i]);
		newObject.transforms[i] = utilityCore::glmMat4ToCudaMat4(transform);
		newObject.inverseTransforms[i] = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
	}
	
        objects.push_back(newObject);
	
	cout << "Loaded " << frameCount << " frames for Object " << objectid << "!" << endl;
        return 1;
    }
}