/* Example N-Body simulation for CIS565 Fall 2013
 * Author: Liam Boone
 * main.cpp */

#include "main.h"

using namespace glm;

#define N_FOR_VIS 10000
#define DT 0.01
#define VISUALIZE 1
//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
	//load geometry
	initGeometry();

	//load mesh
	//bool loadedScene = false;
	//for(int i=1; i<argc; i++){
	//	string header; string data;
	//	istringstream liness(argv[i]);
	//	getline(liness, header, '='); getline(liness, data, '=');
	//	if(strcmp(header.c_str(), "mesh")==0){
	//	  //renderScene = new scene(data);
	//	  mesh = new obj();
	//	  objLoader* loader = new objLoader(data, mesh);
	//	  mesh->buildVBOs();
	//	  delete loader;
	//	  loadedScene = true;
	//	}
	//}

	 // if(!loadedScene){
		//cout << "Usage: mesh=[obj file]" << endl;
		//return 0;
	 // }

    // Launch CUDA/GL

    init(argc, argv);

    cudaGLSetGLDevice( compat_getMaxGflopsDeviceId() );
    initPBO(&pbo);
    cudaGLRegisterBufferObject( planetVBO );
    
#if VISUALIZE == 1
    initCuda(N_FOR_VIS);
#else
    initCuda(2*128);
#endif

    projection = glm::perspective(fovy, float(width)/float(height), zNear, zFar);
    view = glm::lookAt(cameraPosition-glm::vec3(0,0,10), glm::vec3(0), glm::vec3(0,0,1));

    projection = projection * view;

    GLuint passthroughProgram;
    initShaders(program);

    glUseProgram(program[HEIGHT_FIELD]);
    glActiveTexture(GL_TEXTURE0);

    glEnable(GL_DEPTH_TEST);


    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);

    glutMainLoop();

    return 0;
}

void initGeometry(){
	staticGeom geom;
	geom.type = SPHERE;
	geom.rotation = vec3(0,0,0);
	geom.translation = vec3(0,0,-4);
	geom.scale = vec3(10,10,10);
	mat4 transform = utilityCore::buildTransformationMatrix(geom.translation, geom.rotation, geom.scale);
	geom.transform = utilityCore::glmMat4ToCudaMat4(transform);
	geom.inverseTransform = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
	geoms.push_back(geom);
	
	/*staticGeom geom1;
	geom1.type = SPHERE;
	geom1.rotation = vec3(0,0,0);
	geom1.translation = vec3(5,-5,0);
	geom1.scale = vec3(6,6,6);
	transform = utilityCore::buildTransformationMatrix(geom1.translation, geom1.rotation, geom1.scale);
	geom1.transform = utilityCore::glmMat4ToCudaMat4(transform);
	geom1.inverseTransform = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
	geoms.push_back(geom1);*/
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda()
{
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    float4 *dptr=NULL;
    float *dptrvert=NULL;
    cudaGLMapBufferObject((void**)&dptr, pbo);
    cudaGLMapBufferObject((void**)&dptrvert, planetVBO);

	//pack geoms
	staticGeom* gs = new staticGeom[geoms.size()];
	for (int i = 0; i < geoms.size(); i++){
		gs[i] = geoms[i];
	}

	//std::getchar();

    // execute the kernel
    cudaPBFUpdateWrapper(DT, gs, geoms.size());
#if VISUALIZE == 1
    cudaUpdateVBO(dptrvert, field_width, field_height);
#endif
    // unmap buffer object
    cudaGLUnmapBufferObject(planetVBO);
    cudaGLUnmapBufferObject(pbo);
}

int timebase = 0;
int frame = 0;

void display()
{
    static float fps = 0;
    frame++;
    int time=glutGet(GLUT_ELAPSED_TIME);

    if (time - timebase > 1000) {
        fps = frame*1000.0f/(time-timebase);
        timebase = time;
        frame = 0;
    }
    runCuda();

    char title[100];
    sprintf( title, "565 Final Project [%0.2f fps]", fps );
    glutSetWindowTitle(title);

    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, field_width, field_height, 
            GL_RGBA, GL_FLOAT, NULL);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   
#if VISUALIZE == 1
    // VAO, shader program, and texture already bound
    //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    //glDrawElements(GL_TRIANGLES, 6*field_width*field_height,  GL_UNSIGNED_INT, 0);

	glUseProgram(program[HEIGHT_FIELD]);

    glEnableVertexAttribArray(positionLocation);
    glEnableVertexAttribArray(texcoordsLocation);
    
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 

    glBindBuffer(GL_ARRAY_BUFFER, planeTBO);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeIBO);

    glDrawElements(GL_TRIANGLES, 6*field_width*field_height,  GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(positionLocation);
    glDisableVertexAttribArray(texcoordsLocation);

    glUseProgram(program[PASS_THROUGH]);

    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, planetVBO);
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0); 

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planetIBO);
   
    glPointSize(4.0f); 
    glDrawElements(GL_POINTS, N_FOR_VIS+1, GL_UNSIGNED_INT, 0);

    glPointSize(1.0f);

    glDisableVertexAttribArray(positionLocation);

	//mesh
	/*glUseProgram(program[2]);

    glEnableVertexAttribArray(positionLocation);
    glEnableVertexAttribArray(texcoordsLocation);
    
    glBindBuffer(GL_ARRAY_BUFFER, meshVBO);
    glVertexAttribPointer((GLuint)positionLocation, 3, GL_FLOAT, GL_FALSE, 0, 0); 

    glBindBuffer(GL_ARRAY_BUFFER, meshTBO);
    glVertexAttribPointer((GLuint)texcoordsLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshIBO);

	glDrawElements(GL_TRIANGLES, mesh->getVBOsize(),  GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(positionLocation);
    glDisableVertexAttribArray(texcoordsLocation);*/

	//sphere
	glUseProgram(program[2]);

    glEnableVertexAttribArray(positionLocation);
    glEnableVertexAttribArray(texcoordsLocation);
    
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0); 

    glBindBuffer(GL_ARRAY_BUFFER, sphereTBO);
    glVertexAttribPointer((GLuint)texcoordsLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereIBO);

	int num_circles = 25;
	glDrawElements(GL_TRIANGLES, (num_circles-2)*num_circles*6+6*num_circles,  GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(positionLocation);
    glDisableVertexAttribArray(texcoordsLocation);

#endif
    glutPostRedisplay();
    glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y)
{
    std::cout << key << std::endl;
    switch (key) 
    {
        case(27):
			freeCuda();
            exit(1);
            break;
    }
}


//-------------------------------
//----------SETUP STUFF----------
//-------------------------------


void init(int argc, char* argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width, height);
    glutCreateWindow("565 Final Project");

    // Init GLEW
    glewInit();
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        std::cout << "glewInit failed, aborting." << std::endl;
        exit (1);
    }

    initVAO();
    initTextures();
}

void initPBO(GLuint* pbo)
{
    if (pbo) 
    {
        // set up vertex data parameter
        int num_texels = field_width*field_height;
        int num_values = num_texels * 4;
        int size_tex_data = sizeof(GLfloat) * num_values;

        // Generate a buffer ID called a PBO (Pixel Buffer Object)
        glGenBuffers(1,pbo);
        // Make this the current UNPACK buffer (OpenGL is state-based)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
        // Allocate data for the buffer. 4-channel 8-bit image
        glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
        cudaGLRegisterBufferObject( *pbo );
    }
}

void initTextures()
{
    glGenTextures(1,&displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, field_width, field_height, 0, GL_RGBA, GL_FLOAT, NULL);
}

void initVAO(void)
{
    const int fw_1 = field_width-1;
    const int fh_1 = field_height-1;

    int num_verts = field_width*field_height;
    int num_faces = fw_1*fh_1;

    GLfloat *vertices  = new GLfloat[2*num_verts];
    GLfloat *texcoords = new GLfloat[2*num_verts]; 
    GLfloat *bodies    = new GLfloat[4*(N_FOR_VIS+1)];
    GLuint *indices    = new GLuint[6*num_faces];
    GLuint *bindices   = new GLuint[N_FOR_VIS+1];

    glm::vec4 ul(-20.0,-20.0,20.0,20.0);
    glm::vec4 lr(20.0,20.0,0.0,0.0);

    for(int i = 0; i < field_width; ++i)
    {
        for(int j = 0; j < field_height; ++j)
        {
            float alpha = float(i) / float(fw_1);
            float beta = float(j) / float(fh_1);
            vertices[(j*field_width + i)*2  ] = alpha*lr.x + (1-alpha)*ul.x;
            vertices[(j*field_width + i)*2+1] = beta*lr.y + (1-beta)*ul.y;
            texcoords[(j*field_width + i)*2  ] = alpha*lr.z + (1-alpha)*ul.z;
            texcoords[(j*field_width + i)*2+1] = beta*lr.w + (1-beta)*ul.w;
        }
    }

    for(int i = 0; i < fw_1; ++i)
    {
        for(int j = 0; j < fh_1; ++j)
        {
            indices[6*(i+(j*fw_1))    ] = field_width*j + i;
            indices[6*(i+(j*fw_1)) + 1] = field_width*j + i + 1;
            indices[6*(i+(j*fw_1)) + 2] = field_width*(j+1) + i;
            indices[6*(i+(j*fw_1)) + 3] = field_width*(j+1) + i;
            indices[6*(i+(j*fw_1)) + 4] = field_width*(j+1) + i + 1;
            indices[6*(i+(j*fw_1)) + 5] = field_width*j + i + 1;
        }
    }

    for(int i = 0; i < N_FOR_VIS; i++)
    {
        bodies[4*i+0] = 0.0f;
        bodies[4*i+1] = 0.0f;
        bodies[4*i+2] = 0.0f;
        bodies[4*i+3] = 1.0f;
        bindices[i] = i;
    }

    glGenBuffers(1, &planeVBO);
    glGenBuffers(1, &planeTBO);
    glGenBuffers(1, &planeIBO);
    glGenBuffers(1, &planetVBO);
    glGenBuffers(1, &planetIBO);
    
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glBufferData(GL_ARRAY_BUFFER, 2*num_verts*sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, planeTBO);
    glBufferData(GL_ARRAY_BUFFER, 2*num_verts*sizeof(GLfloat), texcoords, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6*num_faces*sizeof(GLuint), indices, GL_STATIC_DRAW);
	
    glBindBuffer(GL_ARRAY_BUFFER, planetVBO);
    glBufferData(GL_ARRAY_BUFFER, 4*(N_FOR_VIS)*sizeof(GLfloat), bodies, GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planetIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS)*sizeof(GLuint), bindices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	//drawMesh();
	drawSphere();

    delete[] vertices;
    delete[] texcoords;
    delete[] bodies;
    delete[] indices;
    delete[] bindices;
}

void initShaders(GLuint * program)
{
    GLint location;
    program[0] = glslUtility::createProgram("shaders/heightVS.glsl", "shaders/heightFS.glsl", attributeLocations, 2);
    glUseProgram(program[0]);
    
    if ((location = glGetUniformLocation(program[0], "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }
    if ((location = glGetUniformLocation(program[0], "u_projMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[0], "u_height")) != -1)
    {
        glUniform1i(location, 0);
    }
    
    program[1] = glslUtility::createProgram("shaders/planetVS.glsl", "shaders/planetGS.glsl", "shaders/planetFS.glsl", attributeLocations, 1);
    glUseProgram(program[1]);
    
    if ((location = glGetUniformLocation(program[1], "u_projMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[1], "u_cameraPos")) != -1)
    {
        glUniform3fv(location, 1, &cameraPosition[0]);
    }

	program[2] = glslUtility::createProgram("shaders/meshVS.glsl", "shaders/meshFS.glsl", attributeLocations, 2);
    glUseProgram(program[2]);
    
    if ((location = glGetUniformLocation(program[2], "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }
    if ((location = glGetUniformLocation(program[2], "u_projMatrix")) != -1)
    {
		 
		glm::mat4 result = projection*utilityCore::cudaMat4ToGlmMat4(geoms[0].transform);
        glUniformMatrix4fv(location, 1, GL_FALSE, &result[0][0]);
    }
    if ((location = glGetUniformLocation(program[2], "u_height")) != -1)
    {
        glUniform1i(location, 0);
    }
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void deletePBO(GLuint* pbo)
{
    if (pbo) 
    {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void shut_down(int return_code)
{
    exit(return_code);
}

void drawMesh(){
    glGenBuffers(1, &meshVBO);
    glGenBuffers(1, &meshTBO);
    glGenBuffers(1, &meshIBO);

    glBindBuffer(GL_ARRAY_BUFFER, meshVBO);
    glBufferData(GL_ARRAY_BUFFER, mesh->getVBOsize()*sizeof(float), mesh->getVBO(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, meshTBO);
	glBufferData(GL_ARRAY_BUFFER, mesh->getVBOsize()*sizeof(float), mesh->getVBO(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->getIBOsize()*sizeof(int), mesh->getIBO(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void drawSphere(){
	int num_circles = 25;
    int num_verts = ((num_circles-1)*num_circles+2)*4;
	int num_indices = (num_circles-2)*num_circles*6+6*num_circles;

    GLfloat *vertices  = new GLfloat[num_verts];
    GLfloat *texcoords = new GLfloat[num_verts]; 
    GLuint *indices    = new GLuint[num_indices];

    //VERTICES AND NORMALS

	float height_angle;
	float radius;
	float y_value;

	vertices[0] = 0;
	vertices[1] = 1;
	vertices[2] = 0;
	vertices[3] = 1;
	texcoords[0] = 0;
	texcoords[1] = 1;
	texcoords[2] = 0;
	texcoords[3] = 1;
	int index = 4;

	for (int i = 1; i < num_circles; i++){
		height_angle = PI/2 - i*(PI/num_circles);
		radius = abs(cos(height_angle));
		y_value = sin(height_angle);
		for (int j = 0; j < num_circles; j++){
			vertices[index] = radius*cos(j*(2*PI/num_circles));
			vertices[index+1] = y_value;
			vertices[index+2] = radius*sin(j*(2*PI/num_circles));
			vertices[index+3] = 1;
			texcoords[index] = radius*cos(j*(2*PI/num_circles));
			texcoords[index+1] = y_value;
			texcoords[index+2] = radius*sin(j*(2*PI/num_circles));
			texcoords[index+3] = 1; 
			index+=4;
		}
	}

	vertices[index] = 0;
	vertices[index+1] = -1;
	vertices[index+2] = 0;
	vertices[index+3] = 1;
	texcoords[index] = 0;
	texcoords[index+1] = -1;
	texcoords[index+2] = 0;
	texcoords[index+3] = 1;

    //INDICES

	//TOP AND BOTTOM
	int x = 1;
	for (int i = 0; i < (num_circles*3); i+=3){
		indices[i] = 0;
		indices[i+1] = x;
		indices[i+2] = x+1;
		x++;
	}
	indices[num_circles*3-1] = 1;
	x = ((num_circles-1)*num_circles+2-1)-1;
	for (int i = num_circles*3; i < num_circles*3*2; i+=3){
		indices[i] = ((num_circles-1)*num_circles+2-1); //last index
		indices[i+1] = x;
		indices[i+2] = x-1;
		x--;
	}
	indices[num_circles*3*2-1] = ((num_circles-1)*num_circles+2-1)-1;

	//MIDDLE SECTION
	x = 1;
	int y = 1+num_circles;
	for (int i = num_circles*3*2; i < (num_circles-2)*num_circles*6+6*num_circles; i+=6){
		indices[i] = x;
		indices[i+1] = x+1;
		indices[i+2] = y;

		indices[i+3] = x+1;
		indices[i+4] = y;
		indices[i+5] = y+1;
		x++;
		y++;
		if (x % num_circles == 1){
			indices[i+4] = x-1;
			indices[i+5] = x-num_circles;
			//std::cout<<x<<" "<<y<<std::endl;
		}

	}

    glGenBuffers(1, &sphereVBO);
    glGenBuffers(1, &sphereTBO);
    glGenBuffers(1, &sphereIBO);
    
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, num_verts*sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, sphereTBO);
    glBufferData(GL_ARRAY_BUFFER, num_verts*sizeof(GLfloat), texcoords, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, num_indices*sizeof(GLuint), indices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	delete[] vertices;
    delete[] texcoords;
    delete[] indices;
}