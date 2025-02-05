from .molecular_base import BaseMolecularDynamics
import openmm as mm
from openmm import app
import openmm.unit as unit
from openmmtools.integrators import VVVRIntegrator

class MolecularTask(BaseMolecularDynamics):
    def setup(self):
        """Setup OpenMM system with specific forcefield"""
        forcefield = app.ForceField("amber99sbildn.xml")
        pdb = app.PDBFile(f"data/{self.cfg.molecule}/start.pdb")
        
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