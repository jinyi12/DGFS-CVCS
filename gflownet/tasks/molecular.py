from .molecular_base import BaseMolecularDynamics
import openmm as mm
from openmm import app
import openmm.unit as unit
from openmmtools.integrators import VVVRIntegrator


class MolecularTask(BaseMolecularDynamics):
    """
    MolecularTask is a subclass of BaseMolecularDynamics that sets up the
    OpenMM system for a specific molecular dynamics simulation.
    """

    def __init__(self, cfg):
        # Store cfg before calling super().__init__
        self.cfg = cfg
        # These need to be set before calling setup() via super().__init__
        self.molecule = cfg.molecule
        self.start_state = cfg.start_state
        self.end_state = cfg.end_state if hasattr(cfg, "end_state") else None
        self.temperature = cfg.temperature
        self.friction = cfg.friction
        self.timestep = cfg.timestep
        # Now call super().__init__ which will use setup()
        super().__init__(cfg)

    def setup(self):
        """Setup OpenMM system with specific forcefield"""
        forcefield = app.ForceField("amber99sbildn.xml")
        pdb = app.PDBFile(f"data/{self.molecule}/{self.start_state}.pdb")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )

        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)

        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])

        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)

        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        return pdb, integrator, simulation, external_force
