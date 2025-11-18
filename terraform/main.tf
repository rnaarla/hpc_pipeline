provider "kubernetes" {
  config_path = "~/.kube/config"
}

# Namespace
resource "kubernetes_namespace" "hpc" {
  metadata {
    name = "hpc"
  }
}

# Prometheus via Helm
resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "prometheus"
  namespace  = kubernetes_namespace.hpc.metadata[0].name
  version    = "19.3.0"
}

# Grafana via Helm
resource "helm_release" "grafana" {
  name       = "grafana"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "grafana"
  namespace  = kubernetes_namespace.hpc.metadata[0].name
  version    = "6.58.4"

  set {
    name  = "adminPassword"
    value = "admin123"
  }
}

# Alertmanager via Helm
resource "helm_release" "alertmanager" {
  name       = "alertmanager"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "alertmanager"
  namespace  = kubernetes_namespace.hpc.metadata[0].name
  version    = "1.7.0"
}
