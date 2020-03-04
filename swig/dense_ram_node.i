%module "dense_ram_node"
%{
#include "../include/dense_ram_node.hpp"
%}

%include "std_vector.i"

namespace std {
  %template(vectorb) vector<bool>;
};

%include "../include/dense_ram_node.hpp"