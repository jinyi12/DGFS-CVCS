import numpy as np
import openmm.unit as unit
from abc import abstractmethod
from .base import BaseTask

class BaseMolecularDynamics(BaseTask):
    """Base class for molecular dynamics tasks in GFlowNet"""
    def __init__(self, cfg):
        super().__init__()
        self.temperature = cfg.temperature * unit.kelvin
        self.friction = cfg.friction / unit.femtoseconds
        self.timestep = cfg.timestep * unit.femtoseconds
        
        # Setup system and get initial state
        self.pdb, self.integrator, self.simulation, self.external_force = self.setup()
        self.get_md_info()
        self.simulation.minimizeEnergy()
        self.position = self.get_state()[0]

    @abstractmethod
    def setup(self):
        """Setup OpenMM system - to be implemented by specific forcefields"""
        pass

    def get_md_info(self):
        """Get system information like masses and thermal noise parameters"""
        self.num_particles = self.simulation.system.getNumParticles()
        m = np.array([
            self.simulation.system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(self.num_particles)
        ])
        self.heavy_atoms = m > 1.1
        self.m = unit.Quantity(m, unit.dalton)
        # ... rest of get_md_info from BaseDynamics

    def step(self, forces):
        """Perform one MD step with given forces"""
        for i in range(forces.shape[0]):
            self.external_force.setParticleParameters(i, i, forces[i])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.step(1)

    def get_state(self):
        """Get current positions and forces"""
        state = self.simulation.context.getState(getPositions=True, getForces=True)
        positions = state.getPositions().value_in_unit(unit.nanometer)
        forces = state.getForces().value_in_unit(
            unit.dalton * unit.nanometer / unit.femtosecond**2
        )
        return positions, forces

    def reset(self):
        """Reset simulation to initial state"""
        for i in range(len(self.position)):
            self.external_force.setParticleParameters(i, i, [0, 0, 0])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.context.setPositions(self.position)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)

    def set_temperature(self, temperature):
        """Update system temperature"""
        self.integrator.setTemperature(temperature * unit.kelvin)

    # Implement BaseTask abstract methods
    def energy(self, x):
        """Compute potential energy for given coordinates"""
        _, potentials = self.energy_function(x)
        return potentials

    def score(self, x):
        """Compute forces for given coordinates"""
        forces, _ = self.energy_function(x)
        return forces

    def log_reward(self, x):
        """Compute log reward as negative energy"""
        return -self.energy(x)

    def energy_function(self, positions):
        """Compute both forces and potential energy"""
        forces, potentials = [], []
        for pos in positions:
            self.simulation.context.setPositions(pos)
            state = self.simulation.context.getState(getForces=True, getEnergy=True)
            force = state.getForces().value_in_unit(
                unit.dalton * unit.nanometer / unit.femtosecond**2
            )
            potential = state.getPotentialEnergy().value_in_unit(
                unit.kilojoules / unit.mole
            )
            forces.append(force)
            potentials.append(potential)
        return np.array(forces), np.array(potentials) 