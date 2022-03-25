
#include <functional>
#include "cubature.h"

#if defined(ENABLE_CUBA)
#include "cuba.h"
#endif

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

    
    // The C definition of the integrand function that is used everywhere
    typedef int (*c_integrand_function) (unsigned ndim, const double *x, void *shared_data, unsigned fdim, double* fval);

    using IntegrandType = potter::Callback<int(unsigned, const double*, void*, unsigned, double*)>;
    
    // The C definition of the Cuba integrand function
    typedef int (*c_Cuba_integrand_function)(const int *ndim, const double x[], const int *ncomp, double f[], void *userdata);

    using CubaIntegrandType = potter::Callback<int(const int *, const double [], const int *, double [], void *)>;

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

#if defined(ENABLE_CUBA)
    template <typename Function>
    struct CubaIntegrandWrapper {
        Function f;
        const Eigen::ArrayXd xmins, xmaxs;
        double Jacobian;

        // No copies allowed
        CubaIntegrandWrapper(const CubaIntegrandWrapper&) = delete;

        /**
         * Note: the integrand function is the standard function that takes x and y in their ranges [a,b]
         */
        CubaIntegrandWrapper(const Function& f, const std::valarray<double> &xmins, const std::valarray<double> &xmaxs) : f(f), 
            xmins(Eigen::Map<const Eigen::ArrayXd>(&(xmins[0]), xmins.size())), 
            xmaxs(Eigen::Map<const Eigen::ArrayXd>(&(xmaxs[0]), xmaxs.size())), 
            Jacobian(1.0/(this->xmaxs-this->xmins).prod()) { };

        // Arguments to this function are as expected by Cuba's C interface
        int call(const int *ndim, const double x[], const int *fdim, double fval[], void *userdata) {
            Eigen::Map<const Eigen::ArrayXd> xmapped(&(x[0]), *ndim);
            // Scale x into the full range
            Eigen::ArrayXd xscaled = xmins + xmapped*xmaxs;
            // Call our standard function, then normalize by Jacobian values
            return f(*ndim, &(xscaled[0]), userdata, *fdim, fval)/Jacobian;
        }
        c_Cuba_integrand_function ptr() {
            potter::CubaIntegrandType::func = std::bind(&CubaIntegrandWrapper<Function>::call, &(*this), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
            return static_cast<potter::c_Cuba_integrand_function>(potter::CubaIntegrandType::callback);
        }
    };
#endif

    template<typename T>
    T value_or(const nlohmann::json& j, const std::string& k, const T &def) { return (j.contains(k)) ? j[k].get<T>() : def; };

    auto get_HCubature_defaults() {
        return nlohmann::json{
            {"fdim", 1},
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
        int fdim = options.at("fdim");
        OutputType val(fdim), err(fdim); val = 0; err = 0;
        
        double reqAbsError = options.at("reqAbsError");
        double reqRelError = options.at("reqRelError");
        int feval_max = options.at("feval_max");

        // Error norm is an enum, so we need to do some casting
        error_norm error_type = static_cast<error_norm>(value_or(options, "norm", static_cast<int>(ERROR_INDIVIDUAL)));

        hcubature(fdim, f, shared, naxes, &xmins[0], &xmaxs[0], feval_max, reqAbsError, reqRelError, error_type, &(val[0]), &(err[0]));
        return std::make_tuple(val, err);
    }

    auto get_VEGAS_defaults() {
        return nlohmann::json{
            {"FDIM", 1}, // The dimension of the output vector
            {"NDIM", 1}, // How many dimensions the integral is being taken over
            {"NVEC", 1},
            {"EPSREL", 1e-8},
            {"EPSABS", 1e-12},
            {"VERBOSE", 0},
            {"LAST", 4},
            {"MINEVAL", 0},
            {"MAXEVAL", 1000000},
            {"NSTART", 1000},
            {"NINCREASE", 500},
            {"NBATCH", 1000},
            {"GRIDNO", 0},
            {"SEED", 0},
        };
    }

    template<typename OutputType>
    auto VEGAS(c_Cuba_integrand_function &integrand, void* shared, const nlohmann::json &options){

        // Allocate buffers for output value(s)
        int FDIM = options.at("FDIM");
        int NDIM = options.at("NDIM");
        int NVEC = options.at("NVEC");
        cubareal EPSREL = options.at("EPSREL");
        cubareal EPSABS = options.at("EPSABS");
        int VERBOSE = options.at("VERBOSE");
        int LAST = options.at("LAST");
        int MINEVAL = options.at("MINEVAL");
        int MAXEVAL = options.at("MAXEVAL");
        int NSTART = options.at("NSTART");
        int NINCREASE = options.at("NINCREASE");
        int NBATCH = options.at("NBATCH");
        int GRIDNO = options.at("GRIDNO");
        int SEED = options.at("SEED");
        
        const char* STATEFILE = nullptr;
        void* SPIN = nullptr;

        int neval, fail;
        std::size_t fdim = static_cast<std::size_t>(FDIM);
        cubareal integral[fdim], error[fdim], prob[fdim]; 
        
        Vegas(FDIM, NDIM, integrand, shared, NVEC,
            EPSREL, EPSABS, VERBOSE, SEED,
            MINEVAL, MAXEVAL, NSTART, NINCREASE, NBATCH,
            GRIDNO, STATEFILE, SPIN,
            &neval, &fail, integral, error, prob
        );
        OutputType val(integral, fdim), err(error, fdim);
        return std::make_tuple(val, err);
    }
}