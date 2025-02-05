document.addEventListener("DOMContentLoaded", function () {
  stage_bba = new NGL.Stage("viewport-bba");
  stage_bba.setParameters({ backgroundColor: "white" });
  stage_bba
    .loadFile("data/bba/path.gro", {
      defaultRepresentation: true,
      asTrajectory: true,
    })
    .then(function (comp) {
      comp.setName("simulation-bba");
      comp.addRepresentation("ball+stick", {
        aspectRatio: 1.0,
        radius: 0.15,
      });
      comp.addTrajectory();
    });

  var toggleSpin_bba = document.getElementById(
    "toggleSpin-bba"
  );
  var isSpinning_bba = false;
  toggleSpin_bba.addEventListener("click", function () {
    if (!isSpinning_bba) {
      stage_bba.setSpin([0, 1, 0], 0.01);
      isSpinning_bba = true;
    } else {
      stage_bba.setSpin(null, null);
      isSpinning_bba = false;
    }
  });

  var toggleRunMDs_bba = document.getElementById(
    "toggleRunMDs-bba"
  );
  var isRunning_bba = false;
  toggleRunMDs_bba.addEventListener("click", function () {
    var trajComp_bba = stage_bba.getComponentsByName(
      "simulation-bba"
    ).list[0].trajList[0];
    var player_bba = new NGL.TrajectoryPlayer(
      trajComp_bba.trajectory,
      { step: 80, mode: "once", interpolateType: "spline", timeout: 200 }
    );
    if (!isRunning_bba) {
      player_bba.play();
      isRunning_bba = true;
    } else {
      player_bba.play();
      player_bba.pause();
      isRunning_bba = false;
    }
  });

  stage_bba_initial = new NGL.Stage("viewport-bba-initial");
  stage_bba_initial.setParameters({ backgroundColor: "white" });
  stage_bba_initial.loadFile("data/bba/unfolded.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  }).then(function (comp) {
    comp.addRepresentation("ball+stick", {
      aspectRatio: 1.0,
      radius: 0.15,
    });
  });

  var toggleSpin_bba_initial = document.getElementById("toggleSpin-bba-inital");
  var isSpinning_bba_initial = false;
  toggleSpin_bba_initial.addEventListener("click", function () {
    if (!isSpinning_bba_initial) {
      stage_bba_initial.setSpin([0, 1, 0], 0.01);
      isSpinning_bba_initial = true;
    } else {
      stage_bba_initial.setSpin(null, null);
      isSpinning_bba_initial = false;
    }
  });

  stage_bba_target = new NGL.Stage("viewport-bba-target");
  stage_bba_target.setParameters({ backgroundColor: "white" });
  stage_bba_target.loadFile("data/bba/folded.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  }).then(function (comp) {
    comp.addRepresentation("ball+stick", {
      aspectRatio: 1.0,
      radius: 0.15,
    });
  });

  var toggleSpin_bba_target = document.getElementById("toggleSpin-bba-target")
  var isSpinning_bba_target = false;
  toggleSpin_bba_target.addEventListener("click", function () {
    if (!isSpinning_bba_target) {
      stage_bba_target.setSpin([0, 1, 0], 0.01);
      isSpinning_bba_target = true;
    } else {
      stage_bba_target.setSpin(null, null);
      isSpinning_bba_target = false;
    }
  });
});