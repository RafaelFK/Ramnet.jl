%module "ram_node"
%{
#include "ram_node.hpp"
%}

%include "std_vector.i"

namespace std {
  %template(vectorb) vector<bool>;
};

%include "ram_node.hpp"