import axios from "axios";

export const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export function mediaUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  const base = API.replace(/\/$/, "");
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

export const client = axios.create({ baseURL: API, timeout: 300000 });
