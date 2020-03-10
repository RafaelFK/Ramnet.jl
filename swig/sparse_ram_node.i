%module "sparse_ram_node"
%{
#include <sparse_ram_node.hpp>
%}

%include "std_vector.i"

namespace std {
  %template(vectorb) vector<bool>;
};

%include "../include/sparse_ram_node.hpp"