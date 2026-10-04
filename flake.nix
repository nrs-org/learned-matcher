{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    git-hooks.url = "github:cachix/git-hooks.nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      systems,
      git-hooks,
      ...
    }:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      checks = forEachSystem (system: {
        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            nixfmt.enable = true;
            statix.enable = true;
            end-of-file-fixer.enable = true;
            trim-trailing-whitespace.enable = true;
            rustfmt.enable = true;
          };
        };
      });

      # libinference.so with the Vulkan title encoder, in the store so it
      # outlives `cargo clean`: `nix build . -o <gcroot>` (see README).
      packages = forEachSystem (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
          inherit (pkgs) lib;
        in
        {
          default = pkgs.rustPlatform.buildRustPackage {
            pname = "inference";
            inherit ((lib.importTOML ./inference/Cargo.toml).package) version;
            src = lib.fileset.toSource {
              root = ./.;
              fileset = lib.fileset.unions [
                ./Cargo.toml
                ./Cargo.lock
                ./inference
              ];
            };
            cargoLock.lockFile = ./Cargo.lock;
            cargoBuildFlags = [
              "-p"
              "inference"
              "--lib"
            ];
            cargoTestFlags = [
              "-p"
              "inference"
              "--lib"
            ];
            buildFeatures = [ "vulkan" ];
            nativeBuildInputs = [ pkgs.pkg-config ];
            buildInputs = [ pkgs.llama-cpp-vulkan ];
          };
        }
      );

      devShells = forEachSystem (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
          inherit (self.checks.${system}.pre-commit-check) shellHook enabledPackages;
        in
        {
          default = pkgs.mkShell {
            inherit shellHook;
            nativeBuildInputs = with pkgs; [ pkg-config ];
            # llama.cpp with its Vulkan backend: only needed to build the
            # inference cdylib with `--features vulkan` (GPU title encoder),
            # found through its llama.pc.
            buildInputs = with pkgs; [ llama-cpp-vulkan ];
            packages =
              enabledPackages
              ++ (with pkgs; [
                sqlite
                rustc
                cargo
                clippy
                nixfmt
                uv
              ]);
          };
        }
      );
    };
}
