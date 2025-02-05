document.addEventListener("DOMContentLoaded", function () {
  stage_trpcage = new NGL.Stage("viewport-trpcage");
  stage_trpcage.setParameters({ backgroundColor: "white" });
  stage_trpcage
    .loadFile("data/trpcage/path.gro", {
      defaultRepresentation: true,
      asTrajectory: true,
    })
    .then(function (comp) {
      comp.setName("simulation-trpcage");
      comp.addRepresentation("ball+stick", {
        aspectRatio: 1.0,
        radius: 0.15,
      });
      comp.addTrajectory();
    });

  var toggleSpin_trpcage = document.getElementById(
    "toggleSpin-trpcage"
  );
  var isSpinning_trpcage = false;
  toggleSpin_trpcage.addEventListener("click", function () {
    if (!isSpinning_trpcage) {
      stage_trpcage.setSpin([0, 1, 0], 0.01);
      isSpinning_trpcage = true;
    } else {
      stage_trpcage.setSpin(null, null);
      isSpinning_trpcage = false;
    }
  });

  var toggleRunMDs_trpcage = document.getElementById(
    "toggleRunMDs-trpcage"
  );
  var isRunning_trpcage = false;
  toggleRunMDs_trpcage.addEventListener("click", function () {
    var trajComp_trpcage = stage_trpcage.getComponentsByName(
      "simulation-trpcage"
    ).list[0].trajList[0];
    var player_trpcage = new NGL.TrajectoryPlayer(
      trajComp_trpcage.trajectory,
      { step: 80, mode: "once", interpolateType: "spline", timeout: 200 }
    );
    if (!isRunning_trpcage) {
      player_trpcage.play();
      isRunning_trpcage = true;
    } else {
      player_trpcage.play();
      player_trpcage.pause();
      isRunning_trpcage = false;
    }
  });

  stage_trpcage_initial = new NGL.Stage("viewport-trpcage-initial");
  stage_trpcage_initial.setParameters({ backgroundColor: "white" });
  stage_trpcage_initial.loadFile("data/trpcage/unfolded.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  }).then(function (comp) {
    comp.addRepresentation("ball+stick", {
      aspectRatio: 1.0,
      radius: 0.15,
    });
  });

  var toggleSpin_trpcage_initial = document.getElementById("toggleSpin-trpcage-inital");
  var isSpinning_trpcage_initial = false;
  toggleSpin_trpcage_initial.addEventListener("click", function () {
    if (!isSpinning_trpcage_initial) {
      stage_trpcage_initial.setSpin([0, 1, 0], 0.01);
      isSpinning_trpcage_initial = true;
    } else {
      stage_trpcage_initial.setSpin(null, null);
      isSpinning_trpcage_initial = false;
    }
  });

  stage_trpcage_target = new NGL.Stage("viewport-trpcage-target");
  stage_trpcage_target.setParameters({ backgroundColor: "white" });
  stage_trpcage_target.loadFile("data/trpcage/folded.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  }).then(function (comp) {
    comp.addRepresentation("ball+stick", {
      aspectRatio: 1.0,
      radius: 0.15,
    });
  });

  var toggleSpin_trpcage_target = document.getElementById("toggleSpin-trpcage-target")
  var isSpinning_trpcage_target = false;
  toggleSpin_trpcage_target.addEventListener("click", function () {
    if (!isSpinning_trpcage_target) {
      stage_trpcage_target.setSpin([0, 1, 0], 0.01);
      isSpinning_trpcage_target = true;
    } else {
      stage_trpcage_target.setSpin(null, null);
      isSpinning_trpcage_target = false;
    }
  });

});