{
  description = "nvinfer_lean_c";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.clang-tools

          # Required for CUDA Runtime API headers
          pkgs.cudaPackages.cuda_cudart
          pkgs.cudaPackages.cuda_nvcc
        ];
      };
    };
}
