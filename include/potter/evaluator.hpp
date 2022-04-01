#pragma once

namespace potter {

    /**
    * This class evaluates the potential energy between a pair of molecules
    * 
    * The molecules are not owned by the class, rather this function builds 
    * functions into which molecules are passed
    */
    template<typename TYPE>
    class PairwisePotentialEvaluator {
    public:
        std::map<std::tuple<std::size_t, std::size_t>, std::function<double(double)> > potential_map;

        ///< Additional contributions, handled in a very generic way via a callback
        std::vector<std::function<double(const Molecule<TYPE>&, const Molecule<TYPE>&)>> generic_contributions;

        /*
        Connect up all the site-site potentials, all molecules have the same number of sites
        */
        void connect_potentials(std::function<double(double)>& f, std::size_t Natoms) {
            for (auto iatom = 0; iatom < Natoms; ++iatom) {
                for (auto jatom = 0; jatom < Natoms; ++jatom) {
                    potential_map[std::make_tuple(iatom, jatom)] = f;
                }
            }
        }

        /**
        * Add a potential energy function for a particular site-site interaction
        * 
        * @param iatom The index of site on molecule A
        * @param jatom The index of site on molecule B
        */
        void add_potential(std::size_t iatom, std::size_t jatom, std::function<double(double)>& f) {
            potential_map[std::make_tuple(iatom, jatom)] = f;
        }

        /**
        * Add a generalized contribution to the overall potential energy for this pair of molecules. Callback function takes two molecules
        */
        void add_generic_contribution(const std::function<double(const Molecule<TYPE>&, const Molecule<TYPE>&)>& f) {
            generic_contributions.push_back(f);
        }

        /**
        * Get a reference to the potential function for a particular site-site interaction
        * 
        * @param i The index of site on molecule A
        * @param j The index of site on molecule B
        */
        auto& get_potential(std::size_t i, std::size_t j) const {
            auto itf = potential_map.find(std::make_tuple(i, j));
            if (itf != potential_map.end()) {
                return itf->second;
            }
            else {
                throw std::invalid_argument("Bad potential");
            }
        }

        /**
        * Actually evaluate the potential energy given the pair of two molecules
        */
        TYPE eval_pot(const Molecule<TYPE>& molA, const Molecule<TYPE>& molB) const {
            TYPE u = 0.0;
            // Sum up site-site contributions to total potential energy
            for (auto iatom = 0; iatom < molA.get_Natoms(); ++iatom) {
                auto xyzA = molA.get_xyz_atom(iatom);
                for (auto jatom = 0; jatom < molB.get_Natoms(); ++jatom) {
                    auto xyzB = molB.get_xyz_atom(jatom);
                    TYPE distij = molA.get_dist(xyzA, xyzB);
                    auto f_potential = get_potential(iatom, jatom);
                    u += f_potential(distij);
                }
            }
            // Add also contributions for other kinds of interactions (e.g., dipole, quadrupole, etc.)
            for (auto contrib : generic_contributions) {
                u += contrib(molA, molB);
            }
            return u;
        }
    };
}
