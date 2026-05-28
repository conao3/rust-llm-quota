{
  description = "llm-quota development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs =
    inputs@{
      flake-parts,
      treefmt-nix,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      perSystem =
        {
          system,
          self',
          ...
        }:
        let
          overlay = final: prev: {
            rustc = prev.rustc;
            cargo = prev.cargo;
            clippy = prev.clippy;
            rustfmt = prev.rustfmt;
            fetchurl =
              args:
              prev.fetchurl (
                if args ? url then
                  let
                    m = builtins.match "https://crates\\.io/api/v1/crates/([^/]+)/([^/]+)/download" args.url;
                  in
                  if m == null then
                    args
                  else
                    args
                    // {
                      url = "https://static.crates.io/crates/${builtins.elemAt m 0}/${builtins.elemAt m 0}-${builtins.elemAt m 1}.crate";
                    }
                else
                  args
              );
          };
          pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ overlay ];
          };
          treefmtEval = treefmt-nix.lib.evalModule pkgs {
            projectRootFile = "flake.nix";
            programs.nixfmt.enable = true;
            programs.rustfmt.enable = true;
          };
        in
        {
          formatter = treefmtEval.config.build.wrapper;

          packages.default = pkgs.rustPlatform.buildRustPackage {
            pname = "llm-quota";
            version = "0.1.0";
            src = ./.;
            cargoLock.lockFile = ./Cargo.lock;
          };

          apps.default = {
            type = "app";
            program = "${pkgs.lib.getExe' self'.packages.default "llm-quota"}";
          };

          devShells.default = pkgs.mkShell {
            packages = with pkgs; [
              rustc
              cargo
              clippy
              rustfmt
            ];
          };
        };
    };
}
