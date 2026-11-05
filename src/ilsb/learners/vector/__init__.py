"""The vector-policy learners: CCIL's trainers v1 and v2, NoiseBC and their batched engine (CCIL#168).

Copied from CCIL at 3b090bb (ccil.training.vector_bc, vector_bc_v2 and noise_bc; ccil.models.engine), with only
the imports changed. The transition identity (identity.py) and the deterministic npz write (io.py) are copies of
CCIL's too. Trainer names and versions, training identities and the identity encoding are unchanged, so a policy
either CCIL trainer wrote loads and evaluates here, a resumable v1 checkpoint resumes here (v2 has no resume), and
the same job trains to the same weights.
"""
