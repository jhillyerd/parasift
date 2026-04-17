{
  pkgs,
  ...
}:
{
  # https://devenv.sh/languages/
  languages.rust = {
    enable = true;
    components = [
      "rustc"
      "cargo"
      "clippy"
      "rustfmt"
      "rust-analyzer"
    ];
  };

  # https://devenv.sh/packages/
  packages = [
    pkgs.openssl
    pkgs.pkg-config
  ];

  # Set required environment variables for OpenSSL
  env = {
    OPENSSL_DIR = "${pkgs.openssl.dev}";
    OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
  };

  # See full reference at https://devenv.sh/reference/options/
}
