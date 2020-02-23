%module "ram_discriminator"
%{
 #include "../include/ram_discriminator.hpp" 
%}

%include "std_vector.i"

namespace std {
  %template(vectorb) vector<bool>;
};

%include "../include/ram_discriminator.hpp"