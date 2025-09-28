//
// Created by Jonathan on 28/09/2025.
//
#include <armadillo>
#include "plotting_utils.h"
#include "../matplotlibcpp.h"
#include "data_types.h"


namespace plt = matplotlibcpp;

void plot_data(arma::mat store_data, ParsedData parsed_thetas, int n, std::string data_type) {
    plt::figure_size(1200, 780);

    std::vector<double> time_vector(parsed_thetas.times.begin(), parsed_thetas.times.end() - 1);

    for (int j = 0; j <= n-1; ++j) {
        for (int i = 6*n-(j+1)*6; i <= 6*n-6*j-1; ++i) {
            std::vector<double> data_row(store_data.row(i).begin(), store_data.row(i).end());
            plt::plot(time_vector, data_row);
            plt::grid(true);
        }

        // Generate title and filename based on data_type
        std::string title, filename;

        if (data_type == "velocity") {
            title = "vel_body_" + std::to_string(n - j);
            filename = "output_folder/velocity_body_" + std::to_string(n - j) + ".png";
        }
        else if (data_type == "acceleration") {
            title = "acc_body_" + std::to_string(n - j);
            filename = "output_folder/acceleration_body_" + std::to_string(n - j) + ".png";
        }
        else if (data_type == "position") {
            title = "pos_body_" + std::to_string(n - j);
            filename = "output_folder/position_body_" + std::to_string(n - j) + ".png";
        }
        else {
            // Default fallback
            title = data_type + "_body_" + std::to_string(n - j);
            filename = "output_folder/" + data_type + "_body_" + std::to_string(n - j) + ".png";
        }

        plt::title(title);
        plt::save(filename);
        plt::clf();
    }
}

//plots velocities - OBS: takes the ParsedData type - set up for specifically 3 bodies (matplotlib does not work well in cpp)
void plot_thetas(ParsedData parsed_thetas,int dofs) {

	// Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);
    // Convert store_velocities.row(0) to std::vector<double>

	// Convert parsed_thetas.times to std::vector<double> (if not already a vector)

	std::vector<double> time_vector(parsed_thetas.times.begin(), parsed_thetas.times.end());

	for (int i = 0; i <= dofs-1; ++i)
	{
		std::vector<double> position_theta(parsed_thetas.pos[i].begin(), parsed_thetas.pos[i].end());
		plt::plot(time_vector, position_theta);
		plt::grid(true);
	}
	//for (int i = 0; i <= 500;i++){
	//std::cout << time_vector[i] << std::endl;
	//}
	std::string title = "positional theta plot";  // Set a single title string
	plt::title(title);  // Now, this will work
	matplotlibcpp::tight_layout();
    matplotlibcpp::save("../output_folder/position_thetas.png");
    plt::clf();

	for (int i = 0; i <= dofs-1; ++i)
	{
		std::vector<double> velocity_theta(parsed_thetas.vel[i].begin(), parsed_thetas.vel[i].end());
		plt::plot(time_vector, velocity_theta);
		plt::grid(true);
	}

	title = "velocity theta plot";  // Set a single title string
	plt::title(title);  // Now, this will work
    plt::save("../output_folder/velocity_thetas.png");
    plt::clf();

    for (int i = 0; i <= dofs-1; ++i)
	{
		std::vector<double> acceleration_theta(parsed_thetas.accel[i].begin(), parsed_thetas.accel[i].end());
		plt::plot(time_vector, acceleration_theta);
		plt::grid(true);
	}

	title = "acceleration theta plot";  // Set a single title string
	plt::title(title);  // Now, this will work
    plt::save("../output_folder/acceleration_thetas.png");
    plt::clf();
}

//plots generalized forces - OBS: takes the ParsedData type
void plot_generalized_forces(arma::mat store_generalized_forces, ParsedData parsed_thetas) {
	// Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);

	// Convert parsed_thetas.times to std::vector<double> (if not already a vector)
	std::vector<double> time_vector(parsed_thetas.times.begin(), parsed_thetas.times.end() - 1);
	for (int i = 0; i <= 2; ++i)
	{
		std::vector<double> generalized_forces(store_generalized_forces.row(i).begin(), store_generalized_forces.row(i).end());
		plt::plot(time_vector, generalized_forces);
		plt::grid(true);
		//std::cout << store_velocities.row(i) << std::endl;
	}
	std::string title = "Generalized_forces";  // Set a single title string
	plt::title(title);  // Now, this will work
    plt::save("output_folder/generalized_forces.png");
    plt::clf();
}