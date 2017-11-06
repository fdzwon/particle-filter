/*
 * particle_filter.cpp
 *
 * 
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	num_particles = 200;
	std::default_random_engine generator;
	std::normal_distribution<double> rand_x(x, std[0]);
	std::normal_distribution<double> rand_y(y, std[1]);
	std::normal_distribution<double> rand_theta(theta, std[2]);
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.x = rand_x(generator);
		p.y = rand_y(generator);
		p.theta = rand_theta(generator);
		p.weight = 1.0;
		p.id = i;
		particles.push_back(p);
		weights.push_back(1.0);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.

	std::default_random_engine generator;
	std::normal_distribution<double> rand_x(0.0, std_pos[0]);
	std::normal_distribution<double> rand_y(0.0, std_pos[1]);
	std::normal_distribution<double> rand_theta(0.0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {

		if (fabs(yaw_rate) < .001) {

			particles[i].x = particles[i].x + velocity*delta_t*(cos(particles[i].theta)) + rand_x(generator);
			particles[i].y = particles[i].y + velocity*delta_t*(sin(particles[i].theta)) + rand_y(generator);
			particles[i].theta = particles[i].theta + yaw_rate*delta_t + rand_theta(generator);

		}
		else {
			particles[i].x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) + rand_x(generator);
			particles[i].y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + rand_y(generator);
			particles[i].theta = particles[i].theta + yaw_rate*delta_t + rand_theta(generator);

		}		

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.

	for (unsigned int i = 0; i < observations.size(); i++) {

		double min = 100000.0;
		for (unsigned int j = 0; j < predicted.size(); j++) {

			double test_min = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if (test_min < min) {

				observations[i].id = predicted[j].id;
				min = test_min;
			}

		}
		
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. 
	// NOTE: The observations are given in the VEHICLE'S coordinate system. The particles are located
	//   according to the MAP'S coordinate system. We transform between the two systems.

	double pi = 3.1415927;

	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	double p1 = 1 / (2 * pi * sig_x * sig_y);
	double p2x = (2 * sig_x * sig_x);
	double p2y = (2 * sig_y * sig_y);

	for (int i = 0; i < num_particles; i++) {

		std::vector<LandmarkObs> predicted;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			//find perspective to each landmark
			double r;
			LandmarkObs pred;
			pred.x = (map_landmarks.landmark_list[j].x_f - particles[i].x) * cos(particles[i].theta) + (map_landmarks.landmark_list[j].y_f - particles[i].y) * sin(particles[i].theta);
			pred.y = (map_landmarks.landmark_list[j].y_f - particles[i].y) * cos(particles[i].theta) - (map_landmarks.landmark_list[j].x_f - particles[i].x) * sin(particles[i].theta);
			pred.id = map_landmarks.landmark_list[j].id_i;
			r = sqrt(pred.x * pred.x + pred.y * pred.y);
			if (r <= sensor_range) {
				 
				predicted.push_back(pred);

			}
	
		}
		dataAssociation(predicted, observations);
		particles[i].weight = 1.0;
		for (unsigned int l = 0; l < observations.size(); l++) {

			for (unsigned int k = 0; k < predicted.size(); k++) {
				if (predicted[k].id == observations[l].id) {

					double obs_x = observations[l].x;
					double obs_y = observations[l].y;
					double p3x = ((predicted[k].x - obs_x)*(predicted[k].x - obs_x))/p2x;
					double p3y = ((predicted[k].y - obs_y)*(predicted[k].y - obs_y))/p2y;
					double p4 = -1.0 * (p3x + p3y);
					double p5 = exp(p4);
					double p = p1*p5;
					particles[i].weight = particles[i].weight*p;
					weights[i] = weights[i] * p;
					}
					
				}
			}

		}

}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 

	std::vector<Particle> particles_new;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::map<int, int> m;
	std::discrete_distribution<> d(weights.begin(), weights.end());
	for (int n = 0; n<num_particles; ++n) {
		Particle particle_res = particles[d(gen)];
		particles_new.push_back(particle_res);
	}
	particles = particles_new;
	for (int i = 0; i < num_particles; i++) {
		weights[i] = particles[i].weight;
	}

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
