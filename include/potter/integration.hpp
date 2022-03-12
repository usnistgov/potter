
#include <functional>
#include "cubature.h"

namespace potter {
    
    // The C definition of the integrand function
    typedef int (*c_integrand_function) (unsigned ndim, const double *x, void *shared_data, unsigned fdim, double* fval);

    auto get_HCubature_defaults() {
        return nlohmann::json{
            {"ndim", 1},
            {"reqAbsError", 0.0},
            {"reqRelError", 1e-13},
            {"feval_max", 1e6 },
        };
    }

    template<typename T>
    T value_or(const nlohmann::json& j, const std::string& k, const T &def) { return (j.contains(k)) ? j.at(k) : def; };

    // Integration with the hcubature algorithm in cubature library
    template<typename OutputType>
    auto HCubature(c_integrand_function &f, void* shared, const std::valarray<double> &xmins, const std::valarray<double> &xmaxs, const nlohmann::json &options){
        int naxes = static_cast<int>(xmins.size());
        
        assert(xmins.size() == xmaxs.size());

        OutputType val = 0.0, err = 0.0;
        int ndim = options.at("ndim"); 
        double reqAbsError = options.at("reqAbsError");
        double reqRelError = options.at("reqRelError");
        int feval_max = options.at("feval_max");
        error_norm error_type = value_or(options, "norm", ERROR_INDIVIDUAL);
        hcubature(ndim, f, shared, naxes, &xmins[0], &xmaxs[0], feval_max, reqAbsError, reqRelError, error_type, &val, &err);
        return std::make_tuple(val, err);
    }

}