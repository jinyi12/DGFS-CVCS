document.addEventListener("DOMContentLoaded", function () {
  stage_aldp = new NGL.Stage("viewport-aldp");
  stage_aldp.setParameters({ backgroundColor: "white" });
  stage_aldp
    .loadFile("data/aldp/path.gro", {
      defaultRepresentation: true,
      asTrajectory: true,
    })
    .then(function (comp) {
      comp.setName("simulation-aldp");
      comp.addRepresentation("ball+stick", {
        aspectRatio: 1.0,
        radius: 0.15,
      });
      comp.addTrajectory();
    });

  var toggleSpin_aldp = document.getElementById(
    "toggleSpin-aldp"
  );
  var isSpinning_aldp = false;
  toggleSpin_aldp.addEventListener("click", function () {
    if (!isSpinning_aldp) {
      stage_aldp.setSpin([0, 1, 0], 0.01);
      isSpinning_aldp = true;
    } else {
      stage_aldp.setSpin(null, null);
      isSpinning_aldp = false;
    }
  });

  var toggleRunMDs_aldp = document.getElementById(
    "toggleRunMDs-aldp"
  );
  var isRunning_aldp = false;
  toggleRunMDs_aldp.addEventListener("click", function () {
    var trajComp_aldp = stage_aldp.getComponentsByName(
      "simulation-aldp"
    ).list[0].trajList[0];
    var player_aldp = new NGL.TrajectoryPlayer(
      trajComp_aldp.trajectory,
      { step: 80, mode: "once", interpolateType: "spline", timeout: 200 }
    );
    if (!isRunning_aldp) {
      player_aldp.play();
      isRunning_aldp = true;
    } else {
      player_aldp.play();
      player_aldp.pause();
      isRunning_aldp = false;
    }
  });

  stage_aldp_initial = new NGL.Stage("viewport-aldp-initial");
  stage_aldp_initial.setParameters({ backgroundColor: "white" });
  stage_aldp_initial.loadFile("data/aldp/c5.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  }).then(function (comp) {
    comp.addRepresentation("ball+stick", {
      aspectRatio: 1.0,
      radius: 0.15,
    });
  });

  var toggleSpin_aldp_initial = document.getElementById("toggleSpin-aldp-inital");
  var isSpinning_aldp_initial = false;
  toggleSpin_aldp_initial.addEventListener("click", function () {
    if (!isSpinning_aldp_initial) {
      stage_aldp_initial.setSpin([0, 1, 0], 0.01);
      isSpinning_aldp_initial = true;
    } else {
      stage_aldp_initial.setSpin(null, null);
      isSpinning_aldp_initial = false;
    }
  });

  stage_aldp_target = new NGL.Stage("viewport-aldp-target");
  stage_aldp_target.setParameters({ backgroundColor: "white" });
  stage_aldp_target.loadFile("data/aldp/c7ax.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  }).then(function (comp) {
    comp.addRepresentation("ball+stick", {
      aspectRatio: 1.0,
      radius: 0.15,
    });
  });

  var toggleSpin_aldp_target = document.getElementById("toggleSpin-aldp-target")
  var isSpinning_aldp_target = false;
  toggleSpin_aldp_target.addEventListener("click", function () {
    if (!isSpinning_aldp_target) {
      stage_aldp_target.setSpin([0, 1, 0], 0.01);
      isSpinning_aldp_target = true;
    } else {
      stage_aldp_target.setSpin(null, null);
      isSpinning_aldp_target = false;
    }
  });
});