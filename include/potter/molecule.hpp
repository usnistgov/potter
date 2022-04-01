#pragma once

namespace potter {
    template <typename TYPE>
    class Molecule {

    public:
        enum class family {not_defined, atomic, linear, nonlinear};
        using EColArray = Eigen::Array<TYPE, Eigen::Dynamic, 1>;
        using CoordMatrix = Eigen::Array<TYPE, 3, Eigen::Dynamic>;
        CoordMatrix coords, coords_initial;

        Molecule(const std::vector<std::vector<TYPE>>& pts) {
            coords.resize(3, pts.size());
            for (auto i = 0; i < pts.size(); ++i) {
                auto& pt = pts[i];
                for (auto j = 0; j < pt.size(); ++j) {
                    coords(j, i) = pt[j];
                }
            }
            coords_initial = coords;
        };
        void reset() {
            coords = coords_initial;
        };

        // Determine the family of the molecule, either atomic, linear, or nonlinear
        family get_family() const{
            if (get_Natoms() == 1){
                return family::atomic;
            }
            else{
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(coords_initial.matrix(), Eigen::ComputeThinU | Eigen::ComputeThinV);
                auto singularValues = svd.singularValues(); // Sorted in decreasing magnitude
                int nonzerovalues = ((singularValues.array()/singularValues.array()(0)).cwiseAbs().eval() > 1e-14).cast<int>().sum();
                // A linear molecule if one non-zero singular value in SVD of coordinates
                return (nonzerovalues == 1) ? family::linear : family::nonlinear;
            }
        }
        
        CoordMatrix rotZ3(TYPE theta) const {
            // Rotation matrix for rotation around the z axis in 3D
            // See https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
            return (Eigen::ArrayXXd(3, 3) <<
                cos(theta), -sin(theta), 0,
                sin(theta), cos(theta), 0,
                0, 0, 1).finished();
        };
        CoordMatrix rotY3(const TYPE theta) const {
            // Rotation matrix for rotation around the y axis in 3D
            // See https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
            const TYPE c = cos(theta), s = sin(theta);
            return (Eigen::ArrayXXd(3, 3) <<
                c, 0, s,
                0, 1, 0,
                -s, 0, c).finished();
        }
        CoordMatrix rotX3(TYPE theta) const {
            // Rotation matrix for rotation around the x axis in 3D
            // See https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
            return (Eigen::ArrayXXd(3, 3) <<
                1, 0, 0,
                0, cos(theta), -sin(theta),
                0, sin(theta), cos(theta)).finished();
        }
        void rotate_plusx(TYPE angle) {
            Eigen::Transform<double, 3, Eigen::Affine> rot(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()));
            coords = rot.linear() * coords.matrix();
            //coords = rotX3(angle).matrix() * coords.matrix(); // Old method
        }
        void rotate_negativex(TYPE angle) {
            rotate_plusx(-angle);
        }
        void rotate_plusy(TYPE angle) {
            Eigen::Transform<double, 3, Eigen::Affine> rot(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()));
            coords = rot.linear() * coords.matrix();
            //coords = rotY3(angle).matrix() * coords.matrix(); // Old method
        }
        void rotate_negativey(TYPE angle) {
            rotate_plusy(-angle);
        }
        void rotate_plusz(TYPE angle) {
            Eigen::Transform<double, 3, Eigen::Affine> rot(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()));
            coords = rot.linear() * coords.matrix();
        }
        void rotate_negativez(TYPE angle) {
            rotate_plusz(-angle);
        }
        void translatex(TYPE dx) {
            coords.row(0) += dx;
        }
        void translatey(TYPE dy) {
            coords.row(1) += dy;
        }

        void translate_3D(TYPE r, TYPE phi, TYPE theta) {
            TYPE p = sin(theta);
            coords.row(0) += r * p * cos(phi);
            coords.row(1) += r * p * sin(phi);
            coords.row(2) += r * cos(theta);
        }

        template<typename ArrayType>
        TYPE get_dist(const ArrayType& rA, const ArrayType& rB) const {
            return sqrt((rA - rB).square().sum());
        }
        Eigen::Index get_Natoms() const {
            return coords.cols();
        }
        auto get_xyz_atom(const Eigen::Index i) const {
            if (i > coords.cols()) {
                throw std::invalid_argument("Bad atom index");
            }
            return coords.col(i);
        }
    };
}