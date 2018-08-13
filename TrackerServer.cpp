/******************************************************************************
This code is a part of the IoTracker system.
It is licensed under the BSD 3-Clause license.
contact: iotracker.info@gmail.com

Copyright (c) 2016, Soliman Nasser , Ibrahim Jubran , Dan Feldman
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    1) Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    2) Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    3) Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/


#include "TrackerServer.h"
#include <fstream>


TrackerServer::TrackerServer()
{
	cout << endl << setup << endl;

	capture_mode = false;
}

TrackerServer::~TrackerServer()
{

}



void TrackerServer::WriteDataToFile()
{
	ofstream myfile1("/iotracker-system/server-node/Input/WayPoints.txt");
	ofstream myfile2("/iotracker-system/server-node/Input/WayPoints_pairIndex.txt");

	for (int i=0; i<WayPoints.size(); i++)
	{
		for (int j=0; j<3; j++)
			myfile1 << (WayPoints.at(i))(j) << endl;

		myfile2 << WayPoints_pairIndex.at(i) << endl;
	}
}


void TrackerServer::computeWayPointLocation()
{
	int cam1_index, cam2_index;
	MatrixXd UV1(NUM_OF_PNTS,2), UV2(NUM_OF_PNTS,2);
	vector<Point2f> points1, points2;
	VectorXd stateVector;
	Vector3d waypoint_temp;
	vector<Vector3d> waypoint_vector;
	vector<int> waypoint_pair_Indicies;

	for (int i=0; i<NUMOFCAMS/2; i++)
	{
		if (!(setup.GetPairActivity(i)))
			continue;

		cam1_index = setup.GetPairIndices(i, 1);
		cam2_index = setup.GetPairIndices(i, 2);

		if (setup.GetAvailability(cam1_index) && setup.GetAvailability(cam2_index))
		{
			points1 = setup.GetPoints(cam1_index);
			points2 = setup.GetPoints(cam2_index);

cout << "points1" << endl << points1 << endl << endl;
cout << "points2" << endl << points2 << endl << endl;

			for (int j=0 ; j<NUM_OF_PNTS ; j++)
			{
				UV1(j,0) = points1.at(j).x;
				UV1(j,1) = points1.at(j).y;
				UV2(j,0) = points2.at(j).x;
				UV2(j,1) = points2.at(j).y;
			}

			stateVector = getS(UV1, UV2, setup.GetCalibMat('K', cam1_index), setup.GetCalibMat('R', cam1_index), setup.GetCalibVec(cam1_index),
				setup.GetCalibMat('K', cam2_index), setup.GetCalibMat('R', cam2_index), setup.GetCalibVec(cam2_index));
cout << "stateVector" << endl << stateVector << endl<< endl;
			waypoint_temp(0) = stateVector(0); waypoint_temp(1) = stateVector(1); waypoint_temp(2) = stateVector(2);

			waypoint_vector.push_back(waypoint_temp);
			waypoint_pair_Indicies.push_back(i);
		}
	}


	float dist;
	float minDist = numeric_limits<float>::max();
	int minPairIndex = 0;
cout << "-------------------------------------------------------------------------------" << endl;
	Vector3d XYZwaypoint;
	for (int i=0; i<waypoint_pair_Indicies.size(); i++)
	{
		XYZwaypoint(0) = 1000* (waypoint_vector.at(i))(1); XYZwaypoint(1) = -1000* (waypoint_vector.at(i))(0); XYZwaypoint(2) = 1000* (waypoint_vector.at(i))(2);
		dist = (setup.GetPairLocation(waypoint_pair_Indicies.at(i)) - XYZwaypoint).norm();	
		
		if (dist < minDist)
		{
			minDist = dist;
			minPairIndex = i;
		}

		////debug
		cout << endl << "XYZwaypoint = " << endl << XYZwaypoint << endl << endl;
		cout << "waypoint_pair_Indicies.at(i) = "  << endl << waypoint_pair_Indicies.at(i) << endl << endl;
		cout << "setup.GetPairLocation(waypoint_pair_Indicies.at(i)= "  << endl <<  setup.GetPairLocation(waypoint_pair_Indicies.at(i)) << endl  << endl << endl;
		cout << "dist= " << endl << dist << endl << endl;
	}
cout << "-------------------------------------------------------------------------------" << endl;
	WayPoints.push_back(waypoint_vector.at(minPairIndex));
	WayPoints_pairIndex.push_back(waypoint_pair_Indicies.at(minPairIndex));

	cout << "waypoint = " << waypoint_vector.at(minPairIndex) << endl;
	cout << "pairIndex = " << waypoint_pair_Indicies.at(minPairIndex) << endl;
}



void TrackerServer::clearCamSetup()
{
	//delete setup;
	setup = CamSetup();
}



void TrackerServer::recieve(int camIndex, vector<Point2f> points)
{
	namedWindow( "Display window", WINDOW_AUTOSIZE );
	char key;

	if (!capture_mode)
	{
		if ((key = cvWaitKey(1)) == 'c')
		{
			if(!mode_mutex.try_lock())
				return;

			capture_mode = true;
			cout << "start capturing mode for " << CAPTURE_PERIOD << "seconds" << endl;
			capture_timer = getTickCount();

			mode_mutex.unlock();
		}
		else if (key == 27)
			{WriteDataToFile();cout << "exit" << endl;}

		return;
	}

	if (((getTickCount() - capture_timer)/getTickFrequency()) < CAPTURE_PERIOD)
	{
		if(!setup_mutex.try_lock())
			return;

		setup.SetPoints(camIndex,points);
		setup.SetAvailability(camIndex,true);
		//cout << "capturing..." << endl;
		cout << "recieved from camIndex = " << camIndex << endl;

		setup_mutex.unlock();
	}
	else
	{
		if(!mode_mutex.try_lock())
			return;

		capture_mode = false;
		cout << "stop capturing mode..." << endl;
		cout << "start computing waypoint location..." << endl << endl;
		computeWayPointLocation();
		clearCamSetup();

		mode_mutex.unlock();
	}

	return;
}




Vector3d TrackerServer::MinDisPoint(Vector3d a, Vector3d d1, Vector3d b, Vector3d d2)
{
	Vector3d w = d1.cross(d2);
	float wdotw = w.dot(w);
	Vector3d bminusa= b - a;

	Vector3d p = a + (((bminusa.cross(d2)).dot(w))/wdotw)*d1;
	Vector3d q = b + (((bminusa.cross(d1)).dot(w))/wdotw)*d2;

	Vector3d m = (p+q)/2;

	return m;
}




Vector3d TrackerServer::ComputeLocation(Matrix2d M, Matrix3d K1, Matrix3d R1, Vector3d T1, Matrix3d K2, Matrix3d R2, Vector3d T2)
{
	Vector3d d1; d1 << M(0,0),M(0,1),K1(0,0);
	Vector3d temp; temp << K1(0,2),K1(1,2),0;
	d1 = d1 - temp;
	d1 = R1*(d1);//.transpose());

	Vector3d d2; d2 << M(1,0),M(1,1),K2(0,0);
	temp << K2(0,2),K2(1,2),0;
	d2 = d2 - temp;
	d2 = R2*(d2);//.transpose());

	Vector3d result;
	result = MinDisPoint(R1*(T1), d1, R2*(T2), d2);
	float t = -result(1);
	result(1) = result(0);
	result(0) = t;

	return result;
}





VectorXd TrackerServer::getS(MatrixXd UV1, MatrixXd UV2, Matrix3d K1, Matrix3d R1, Vector3d T1, Matrix3d K2, Matrix3d R2, Vector3d T2)
{

	stringstream xyz_points;

	//cout << endl<< endl<< endl<< "UV1:" <<endl<<UV1  << endl;
	//cout <<  "UV2:" <<endl<<UV2  << endl;
	//determine which Calibration matrices to use
	Matrix2d M;
	vector<Vector3d> XYZ;
	Vector3d xyz;
	MatrixXd RPYmat(NUM_OF_PNTS,3);

	for (int i = 0; i < NUM_OF_PNTS; i++)
	{
		M(0,0) = UV1(i,0);
		M(0,1) = UV1(i,1);
		M(1,0) = UV2(i,0);
		M(1,1) = UV2(i,1);
//cout <<endl << "M:" << M << endl;
		xyz = ComputeLocation(M,K1,R1,T1,K2,R2,T2);
		RPYmat(i,0) = xyz(0)/1000;
		RPYmat(i,1) = xyz(1)/1000;
		RPYmat(i,2) = xyz(2)/1000;
		XYZ.push_back(xyz/1000);
	//cout << "xyz"<<i+1 << ": " 	<< xyz << endl;

	
	}

	Vector3d XYZmean; XYZmean << 0,0,0;
	for (int i = 0; i < NUM_OF_PNTS; i++)
	{
		XYZmean += XYZ.at(i);
	}
	XYZmean /= NUM_OF_PNTS;

	VectorXd result(3);
	result << XYZmean;//,RPY;

	//cout << "Result: " << result << endl << endl << endl;




	return result;
}

