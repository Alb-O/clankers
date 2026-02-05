{
  nightlyDate = "2026-01-21";
  targets = [ ];
  devShell.extraPackages = [ ];

  # Base build config - inherited by all packages unless overridden
  build.pname = "clankers";
  build.doCheck = false;
  build.postInstall = "";
  build.cargoOutputHashes = { };

  # Multi-package support
  packages = { };
}
