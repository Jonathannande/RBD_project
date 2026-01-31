
//
// Created by Jonathan on 28/09/2025.
//

#pragma once
#include <armadillo>
#include <string>
#include <variant>

class Hinge_map {
private:
public:
  virtual void get_animation(
      double theta) = 0; // might need to use another type for this one;

  // different type of coriolis computation based on hinge type
  virtual arma::vec6 compute_coriolis(const arma::vec &theta,
                                      const arma::vec &theta_dot,
                                      const arma::vec &velocity_vector) = 0;
  virtual arma::vec6 compute_coriolis(const arma::vec &theta_dot,
                                      const arma::vec &velocity_vector) = 0;

  virtual arma::vec6 get_hinge_map() = 0;
  virtual arma::mat::fixed<2, 6> get_hinge_map(const arma::vec &theta);

  virtual arma::vec6 get_transposed_hinge_map() = 0;
  virtual arma::mat::fixed<6, 2>
  get_transposed_hinge_map(const arma::vec &theta) = 0;

  virtual arma::vec6 compute_hinge_state(const arma::vec &theta);
};

class Rotational : public Hinge_map {
  Rotational(arma::mat global_frame) {
    arma::vec6 hinge_map_ = {1, 0, 0, 0, 0, 0};
    hinge_map = global_frame * hinge_map_;
    transpose_hinge_map = hinge_map.t();
  };

private:
  arma::vec6 hinge_map;
  arma::vec6 transpose_hinge_map;
  bool is_dependent = false;

public:
  void get_animation(double theta) override;
  arma::vec6 compute_coriolis(const arma::vec &theta_dot,
                              const arma::vec &velocity_vector) override;
  arma::vec6 get_hinge_map() override;
  arma::vec6 get_transposed_hinge_map() override;
  arma::vec6 computute_hinge_state(const arma::vec &theta);
};

class Translational : public Hinge_map {
private:
  arma::vec6 hinge_map;
  bool is_depeendent = false;

public:
  Translational(arma::mat global_frame) {
    arma::vec6 hinge_map_ = {0, 0, 0, 1, 0, 0};
    hinge_map = global_frame * hinge_map_;
  }
  void get_animation(double theta) override;
  arma::vec6 compute_coriolis(const arma::vec &theta,
                              const arma::vec &theta_dot,
                              const arma::vec &velocity_vector) override;
  arma::vec6 get_hinge_map() override;
  arma::vec6 get_transposed_hinge_map() override;
  arma::vec6 computute_hinge_state(const arma::vec &theta);
};

class Cylindrical : public Hinge_map {
private:
  arma::vec6 hinge_map;
  bool is_depeendent = false;

public:
  Cylindrical(arma::mat global_frame) {

    arma::mat::fixed<2, 6> hinge_map_ = arma::zeros(2, 6);
    hinge_map_(0, 0) = 1;
    hinge_map_(3, 1) = 1;
    hinge_map = global_frame * hinge_map_;
  }
  void get_animation(double theta) override;
  arma::vec6 compute_coriolis(const arma::vec &theta,
                              const arma::vec &theta_dot,
                              const arma::vec &velocity_vector) override;
  arma::mat::fixed<2, 6> get_hinge_map(const arma::vec &theta) override;
  arma::mat::fixed<6, 2>
  get_transposed_hinge_map(const arma::vec &theta) override;
  arma::vec6 computute_hinge_state(const arma::vec &theta);
};

class Skrew : public Hinge_map {
private:
  arma::vec6 hinge_map;
  bool is_depeendent = false;

public:
  Skrew(arma::mat global_frame, double gear_ratio) {

    arma::mat::fixed<2, 6> hinge_map_ = arma::zeros(1, 6);
    hinge_map_(0, 0) = 1;
    hinge_map_(3, 0) = gear_ratio;
    hinge_map = global_frame * hinge_map_;
  }
  void get_animation(double theta) override;
  arma::vec6 compute_coriolis(const arma::vec &theta,
                              const arma::vec &theta_dot,
                              const arma::vec &velocity_vector) override;
  arma::mat::fixed<2, 6> get_hinge_map(const arma::vec &theta) override;
  arma::mat::fixed<6, 2>
  get_transposed_hinge_map(const arma::vec &theta) override;
  arma::vec6 computute_hinge_state(const arma::vec &theta);
};

class Universal : public Hinge_map {
private:
  arma::vec6 hinge_map;
  bool is_depeendent = false;

public:
  Universal(arma::mat global_frame) {

    arma::mat::fixed<2, 6> hinge_map_ = arma::zeros(2, 6);
    hinge_map = global_frame * hinge_map_;
  }
  void get_animation(double theta) override;
  arma::vec6 compute_coriolis(const arma::vec &theta,
                              const arma::vec &theta_dot,
                              const arma::vec &velocity_vector) override;
  arma::mat::fixed<2, 6> get_hinge_map(const arma::vec &theta) override;
  arma::mat::fixed<6, 2>
  get_transposed_hinge_map(const arma::vec &theta) override;
  arma::vec6 computute_hinge_state(const arma::vec &theta);
};
