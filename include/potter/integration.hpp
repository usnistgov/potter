
#include <functional>
#include "cubature.h"

namespace potter {

    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    
    // See https://blog.mbedded.ninja/programming/languages/c-plus-plus/callbacks/ (option 6)
    // See https://replit.com/@gbmhunter/c-callback-in-cpp-using-templating-functional-bind#main.cpp

    template <typename T> struct Callback;

    template <typename Ret, typename... Params>
    struct Callback<Ret(Params...)> {
        template <typename... Args>
        static Ret callback(Args... args) {
            return func(args...);
        }
        static std::function<Ret(Params...)> func;
    };

    template <typename Ret, typename... Params>
    std::function<Ret(Params...)> Callback<Ret(Params...)>::func;

    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------

    
    // The C definition of the integrand function
    typedef int (*c_integrand_function) (unsigned ndim, const double *x, void *shared_data, unsigned fdim, double* fval);

    using IntegrandType = potter::Callback<int(unsigned, const double*, void*, unsigned, double*)>;

    template <typename Function>
    struct IntegrandWrapper {
        Function f;
        IntegrandWrapper(const IntegrandWrapper&) = delete;

        IntegrandWrapper(const Function& f) : f(f) {};
        int call(unsigned ndim, const double* x, void* shared_data, unsigned fdim, double* fval) {
            return f(ndim, x, shared_data, fdim, fval);
        }
        c_integrand_function ptr() {
            potter::IntegrandType::func = std::bind(&IntegrandWrapper<Function>::call, &(*this), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
            return static_cast<potter::c_integrand_function>(potter::IntegrandType::callback);
        }
    };

    template<typename T>
    T value_or(const nlohmann::json& j, const std::string& k, const T &def) { return (j.contains(k)) ? j.at(k) : def; };

    auto get_HCubature_defaults() {
        return nlohmann::json{
            {"ndim", 1},
            {"reqAbsError", 0.0},
            {"reqRelError", 1e-13},
            {"feval_max", 1e6 },
        };
    }

    // Integration with the hcubature algorithm in cubature library
    template<typename OutputType>
    auto HCubature(c_integrand_function &f, void* shared, const std::valarray<double> &xmins, const std::valarray<double> &xmaxs, const nlohmann::json &options){
        int naxes = static_cast<int>(xmins.size());
        
        assert(xmins.size() == xmaxs.size());

        // Allocate buffers for output value(s)
        int ndim = options.at("ndim");
        OutputType val(ndim), err(ndim); val = 0; err = 0;
        
        double reqAbsError = options.at("reqAbsError");
        double reqRelError = options.at("reqRelError");
        int feval_max = options.at("feval_max");
        error_norm error_type = value_or(options, "norm", ERROR_INDIVIDUAL);

        hcubature(ndim, f, shared, naxes, &xmins[0], &xmaxs[0], feval_max, reqAbsError, reqRelError, error_type, &(val[0]), &(err[0]));
        return std::make_tuple(val, err);
    }

    

}