
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

    struct CubaCallArgs{
        c_integrand_function func;
        void *userdata;
        const std::valarray<double> &xmins, &xmaxs;
        double Jacobian;
        
        CubaCallArgs(c_integrand_function func, void *userdata, const std::valarray<double> &xmins, const std::valarray<double> &xmaxs) 
        : func(func), userdata(userdata), xmins(xmins), xmaxs(xmaxs) {
            Eigen::Map<const Eigen::ArrayXd> xmins_(&(xmins[0]), xmins.size());
            Eigen::Map<const Eigen::ArrayXd> xmaxs_(&(xmaxs[0]), xmaxs.size());    
            Jacobian = 1.0/(xmaxs_-xmins_).prod();
        }
    };

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
    auto VEGAS(c_integrand_function &integrand, void* userdata, const std::valarray<double> &xmins, const std::valarray<double> &xmaxs, const nlohmann::json &options){

        // Allocate buffers for output value(s)
        int FDIM = options.at("FDIM");
        int XDIM = options.at("NDIM");
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

        // Argument size checking
        if (xmins.size() != xmaxs.size()){
            throw std::invalid_argument("lengths of xmin and xmax don't match");
        }
        if (xmins.size() != XDIM){
            throw std::invalid_argument("NDIM doesn't match length of x");
        }

        // The function that remaps from Cuba to integrand function style
        CubaCallArgs args(integrand, userdata, xmins, xmaxs);
        auto Cuba_integrand = [](const int *pndim, const cubareal *x, const int *fdim, cubareal *fval, void *p_shared_data) -> int{
            auto& shared = *((class CubaCallArgs*)(p_shared_data));
            Eigen::Map<const Eigen::ArrayXd> xmapped(&(x[0]), *pndim);
            Eigen::Map<const Eigen::ArrayXd> xmins(&(shared.xmins[0]), *pndim);
            Eigen::Map<const Eigen::ArrayXd> xmaxs(&(shared.xmaxs[0]), *pndim);
            // Scale x from [0,1] into the full range [xmin, xmax]
            Eigen::ArrayXd xunscaled = xmins + xmapped*(xmaxs-xmins);
            const unsigned ndim_unsigned = *pndim, fdim_unsigned = *fdim; 
            // Call our standard function
            shared.func(ndim_unsigned, &(xunscaled[0]), shared.userdata, fdim_unsigned, fval);
            // Then normalize by Jacobian values
            Eigen::Map<Eigen::ArrayXd> fmapped(&(fval[0]), *fdim);
            fmapped /= shared.Jacobian;
            return 0; // success
        };
        
        int neval, fail;
        std::size_t fdim = static_cast<std::size_t>(FDIM);
        cubareal integral[fdim], error[fdim], prob[fdim]; 
        Vegas(XDIM, FDIM, Cuba_integrand, &args, NVEC,
            EPSREL, EPSABS, VERBOSE, SEED,
            MINEVAL, MAXEVAL, NSTART, NINCREASE, NBATCH,
            GRIDNO, STATEFILE, SPIN,
            &neval, &fail, integral, error, prob
        );
        OutputType val(FDIM), err(FDIM);
        Eigen::Map<Eigen::ArrayXd>(&(val[0]), FDIM) = Eigen::Map<Eigen::ArrayXd>(&(integral[0]), FDIM);
        Eigen::Map<Eigen::ArrayXd>(&(err[0]), FDIM) = Eigen::Map<Eigen::ArrayXd>(&(error[0]), FDIM);
        return std::make_tuple(val, err);
    }
}