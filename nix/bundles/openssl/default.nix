{ pkgs, ... }:
{
  __outputs.perSystem.buildDeps.openssl = {
    buildInputs = [ pkgs.openssl ];
    nativeBuildInputs = [ pkgs.pkg-config ];
  };

  __outputs.perSystem.devShells.openssl = pkgs.mkShell {
    nativeBuildInputs = [ pkgs.pkg-config ];
    buildInputs = [ pkgs.openssl ];
  };
}
