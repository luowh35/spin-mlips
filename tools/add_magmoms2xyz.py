#!/usr/bin/env python3
import numpy as np
import math
import sys

DEBUG_RANDOM = True
FIXED_RANDOM_SEED = 20240527
RNG_SEED = FIXED_RANDOM_SEED if DEBUG_RANDOM else None

# =========================================================


# ================== random vector generate ==================

def generate_unit_vectors_within_angle(n, max_angle_deg, seed=None):
    if seed is not None:
        np.random.seed(seed)

    max_angle_rad = math.radians(max_angle_deg)
    cos_max = math.cos(max_angle_rad)

    vectors = np.zeros((n, 3))
    for i in range(n):
        phi = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(cos_max, 1.0)
        r = math.sqrt(1 - z**2)
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        vectors[i] = [x, y, z]
    return vectors


def generate_random_unit_vectors(n, seed=None):
    if seed is not None:
        np.random.seed(seed)

    v = np.random.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v


# ================== extendxyz read ==================

def read_xyz_raw(filename):
    """Read line by line"""
    with open(filename) as f:
        lines = [l.rstrip("\n") for l in f]

    natoms = int(lines[0].strip())
    comment = lines[1]
    atom_lines = lines[2:2 + natoms]

    return natoms, comment, atom_lines


def update_properties_line(comment):
    """
    If extendxyz, add magnetize_home: R: 3 in Properties
    """
    if "Properties=" not in comment:
        return comment

    if "magnetic_moment" in comment:
        return comment

    return comment.replace(
        "Properties=species:S:1:pos:R:3",
        "Properties=species:S:1:pos:R:3:magnetic_moment:R:3"
    )


def write_xyz(filename, natoms, comment, atom_lines):
    with open(filename, "w") as f:
        f.write(f"{natoms}\n")
        f.write(comment + "\n")
        for l in atom_lines:
            f.write(l + "\n")


# ================== main ==================

def main():
    if len(sys.argv) != 3:
        print("Usage: python add_magmoms_to_xyz.py input.xyz output.xyz")
        sys.exit(1)

    infile, outfile = sys.argv[1], sys.argv[2]
    natoms, comment, atom_lines = read_xyz_raw(infile)

    print(f"Read {natoms} atoms")

    # ---- refresh extendxyz header ----
    comment = update_properties_line(comment)

    # ---- 交互输入 ----
    mag_elems = input(
        "Please enter magnetic elements (comma separated, such as Cr, Fe; Enter to indicate none):"
    ).strip()
    mag_elements = {e.strip() for e in mag_elems.split(",")} if mag_elems else set()

    mag_value = float(
        input("Please enter the magnetic moment of the magnetic atom (μB)：")
    )

    mode = input(
        "rand mode [full/cone] (default: cone)："
    ).strip() or "cone"

    max_angle = 0.0
    if mode == "cone":
        max_angle = float(
            input("Please enter the maximum inclination angle θ_max (deg)：")
        )

    # ---- Find magnetic atoms ----
    elements = [l.split()[0] for l in atom_lines]
    mag_indices = [i for i, e in enumerate(elements) if e in mag_elements]
    nmag = len(mag_indices)

    print(f"Magnetic atoms: {nmag}")

    # ---- Magnetic moments generate ----
    if nmag > 0:
        if mode == "full":
            dirs = generate_random_unit_vectors(nmag, seed=RNG_SEED)
        else:
            dirs = generate_unit_vectors_within_angle(
                nmag, max_angle, seed=RNG_SEED
            )
        mag_vectors = dirs * mag_value
    else:
        mag_vectors = np.zeros((0, 3))

    # ---- Atomic line literal inheritance and write magnetic moment ----
    out_atom_lines = []
    mag_counter = 0

    for i, line in enumerate(atom_lines):
        if i in mag_indices:
            mx, my, mz = mag_vectors[mag_counter]
            mag_counter += 1
            out_atom_lines.append(f"{line} {mx:.3f} {my:.3f} {mz:.3f}")
        else:
            out_atom_lines.append(f"{line} 0.000 0.000 0.000")

    write_xyz(outfile, natoms, comment, out_atom_lines)
    print(f"Written: {outfile}")


if __name__ == "__main__":
    main()
