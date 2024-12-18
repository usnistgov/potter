# potter

potter is a set of computational routines to:

* Evaluate virial coefficients of generic potentials and rigid molecules from adaptive multidimensional integration
* Utilize closed-form and empirical routines developed for various canonical potentials (work in progress)

The C++ API documentation (generated by [doxygen](http://www.doxygen.nl/) ) is [available here](potter-1.0-doxygen.pdf)

Automated Tests: [![build and run Catch tests](https://github.com/usnistgov/potter/actions/workflows/runcatch.yml/badge.svg)](https://github.com/usnistgov/potter/actions/workflows/runcatch.yml)

Brought to you by:

* Ian Bell, NIST, ian.bell@nist.gov
* Sven Pohl, Ruhr-Universitat Bochum

## License

MIT licensed

## Dependencies

* Unmodified [Eigen](https://eigen.tuxfamily.org/dox/) for matrix operations
* Unmodified [ThreadPool2](https://github.com/stfx/ThreadPool2) for thread pooling
* Unmodified [cubature](https://github.com/stevengj/cubature) for multidimensional integration
* Unmodified [Cuba](http://www.feynarts.de/cuba) for multidimensional integration

## Contributing/Getting Help

If you would like to contribute to ``potter`` or report a problem, please open a pull request or submit an issue.  Especially welcome would be additional tests.  

If you want to discuss or request assistance, please open an issue.

## Installation

### Prerequisites

You will need:

* cmake (on windows, install from cmake, on linux ``sudo apt install cmake`` should do it, on OSX, ``brew install cmake``)
* a compiler (on windows, Visual Studio 2015+ (express version is fine), g++ on linux/OSX)

### Cmake build

Starting in the root of the repo (a debug build with the default compiler, here on linux):

``` 
git clone --recursive https://github.com/usnistgov/potter
cd potter
mkdir build
cd build
cmake ..
cmake --build .
```
For Visual Studio 2015 (64-bit) in release mode, you would do:
``` 
git clone --recursive https://github.com/usnistgov/potter
cd potter
mkdir build
cd build
cmake .. -G "Visual Studio 14 2015 Win64"
cmake --build . --config Release
```

For other options, see the cmake docs

If you need to update your submodules
```
git submodule update --init
```

## Debugging

* ``lstopo`` from the hwloc package can tell you the physical configuration of the cores
* ``taskset --cpu-list 0-23 nohup ./potter &`` will run on the first 24 threads (or you could split up in a different way)
