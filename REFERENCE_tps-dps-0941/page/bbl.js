document.addEventListener("DOMContentLoaded", function () {
  stage_bbl = new NGL.Stage("viewport-bbl");
  stage_bbl.setParameters({ backgroundColor: "white" });
  stage_bbl
    .loadFile("data/bbl/path.gro", {
      defaultRepresentation: true,
      asTrajectory: true,
    })
    .then(function (comp) {
      comp.setName("simulation-bbl");
      comp.addRepresentation("ball+stick", {
        aspectRatio: 1.0,
        radius: 0.15,
      });
      comp.addTrajectory();
    });

  var toggleSpin_bbl = document.getElementById(
    "toggleSpin-bbl"
  );
  var isSpinning_bbl = false;
  toggleSpin_bbl.addEventListener("click", function () {
    if (!isSpinning_bbl) {
      stage_bbl.setSpin([0, 1, 0], 0.01);
      isSpinning_bbl = true;
    } else {
      stage_bbl.setSpin(null, null);
      isSpinning_bbl = false;
    }
  });

  var toggleRunMDs_bbl = document.getElementById(
    "toggleRunMDs-bbl"
  );
  var isRunning_bbl = false;
  toggleRunMDs_bbl.addEventListener("click", function () {
    var trajComp_bbl = stage_bbl.getComponentsByName(
      "simulation-bbl"
    ).list[0].trajList[0];
    var player_bbl = new NGL.TrajectoryPlayer(
      trajComp_bbl.trajectory,
      { step: 80, mode: "once", interpolateType: "spline", timeout: 200 }
    );
    if (!isRunning_bbl) {
      player_bbl.play();
      isRunning_bbl = true;
    } else {
      player_bbl.play();
      player_bbl.pause();
      isRunning_bbl = false;
    }
  });

  stage_bbl_initial = new NGL.Stage("viewport-bbl-initial");
  stage_bbl_initial.setParameters({ backgroundColor: "white" });
  stage_bbl_initial.loadFile("data/bbl/unfolded.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  }).then(function (comp) {
    comp.addRepresentation("ball+stick", {
      aspectRatio: 1.0,
      radius: 0.15,
    });
  });

  var toggleSpin_bbl_initial = document.getElementById("toggleSpin-bbl-inital");
  var isSpinning_bbl_initial = false;
  toggleSpin_bbl_initial.addEventListener("click", function () {
    if (!isSpinning_bbl_initial) {
      stage_bbl_initial.setSpin([0, 1, 0], 0.01);
      isSpinning_bbl_initial = true;
    } else {
      stage_bbl_initial.setSpin(null, null);
      isSpinning_bbl_initial = false;
    }
  });

  stage_bbl_target = new NGL.Stage("viewport-bbl-target");
  stage_bbl_target.setParameters({ backgroundColor: "white" });
  stage_bbl_target.loadFile("data/bbl/folded.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  }).then(function (comp) {
    comp.removeRepresentation("ball+stick");
    comp.addRepresentation("ball+stick", {
      aspectRatio: 1.0,
      radius: 0.1,
    });
    comp.addRepresentation("ribbon", {
      colorScheme: "protein"
    });
  });

  var toggleSpin_bbl_target = document.getElementById("toggleSpin-bbl-target")
  var isSpinning_bbl_target = false;
  toggleSpin_bbl_target.addEventListener("click", function () {
    if (!isSpinning_bbl_target) {
      stage_bbl_target.setSpin([0, 1, 0], 0.01);
      isSpinning_bbl_target = true;
    } else {
      stage_bbl_target.setSpin(null, null);
      isSpinning_bbl_target = false;
    }
  });
});